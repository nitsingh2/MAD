#!/bin/bash
# Tiered prefix caching — KV offload overlay (orthogonal to CONNECTOR/WIDE_EP).
# Layers a CPU-RAM KV tier (GPU KV -> host) on top of the disagg P/D connector via
# vLLM's MultiConnector. Load reads the first matching sub-connector, save writes to
# all — the offload sub-connector is listed first so a decode worker hits the local
# CPU cache before the P->D fetch. KV_OFFLOAD=none is a no-op.
#
# The CPU tier's backend is selectable (OFFLOAD_BACKEND):
#   SimpleCPUOffloadConnector  (default) -> Hash-driven store path, no paired-array
#                                           invariant. CPU tier sized by OFFLOAD_CPU_BYTES.
#                                           No filesystem tier support.
#   OffloadingConnector                  -> Strided block-id store path; known to assert
#                                           under MoRIIO write-mode disagg on warm reuse
#                                           (_build_store_jobs invariant). Use for
#                                           testing/comparison only. CPU tier sized by
#                                           OFFLOAD_CPU_BYTES.
#   LMCacheConnectorV1                   -> LMCache CPU tier sized by
#                                           LMCACHE_MAX_LOCAL_CPU_SIZE. Requires an image
#                                           with `lmcache` installed (the stock
#                                           vllm/vllm-openai-rocm image does NOT ship it);
#                                           LMCache reads its config from the process env.
#
# The LMCacheConnectorV1 backend can add a filesystem tier below the CPU tier via
# OFFLOAD_DISK_PATH (GPU -> CPU RAM -> disk): LMCache (LRU) spills CPU-evicted chunks to
# its local_disk. A per-host subdir is appended so the prefill and decode nodes don't
# collide on a shared mount. The other backends do not support a disk tier.
#
# MoRIIO compatibility (CURRENT, 2026-09): only the NATIVE CPU-offload backends
# (SimpleCPUOffloadConnector / OffloadingConnector) work under MoRIIO disagg, and
# ONLY in READ mode (set MORIIO_READ_MODE=1). MoRIIO write mode + OffloadingConnector
# deadlocks/asserts on the warm-reuse store path. LMCacheConnectorV1 does NOT work
# with MoRIIO anywhere: MultiConnector[LMCache, MoRIIO] hits a GPU memory-access
# fault during the warmup forward (unresolved host-side registration race). Use
# LMCacheConnectorV1 only under the NIXL/rixl connector, not MoRIIO.
#
# Env: KV_OFFLOAD        = none (default) | cpu
#      OFFLOAD_BACKEND   = SimpleCPUOffloadConnector (default) | OffloadingConnector | LMCacheConnectorV1
#                          (only read when KV_OFFLOAD=cpu)
#      OFFLOAD_CPU_BYTES = pinned host bytes for SimpleCPUOffloadConnector/OffloadingConnector (default 100 GB)
#      LMCACHE_MAX_LOCAL_CPU_SIZE  = per-worker CPU tier in GB for LMCacheConnectorV1 (default 100.0)
#      OFFLOAD_DISK_PATH           = base dir for a filesystem tier (unset = no disk tier).
#                                    LMCacheConnectorV1 backend only. Node-local disk recommended.
#      LMCACHE_MAX_LOCAL_DISK_SIZE = LMCacheConnectorV1 only: per-worker disk tier in GB (default 0.0)

KV_OFFLOAD="${KV_OFFLOAD:-none}"
OFFLOAD_BACKEND="${OFFLOAD_BACKEND:-SimpleCPUOffloadConnector}"
export OFFLOAD_CPU_BYTES="${OFFLOAD_CPU_BYTES:-107374182400}"

kv_offload_enabled() {
    [[ "${KV_OFFLOAD:-none}" != "none" ]]
}

# Per-host dir for a filesystem tier; empty when OFFLOAD_DISK_PATH is unset.
_kv_offload_fs_dir() {
    [[ -n "${OFFLOAD_DISK_PATH:-}" ]] || return 0
    printf '%s/%s' "${OFFLOAD_DISK_PATH%/}" "$(hostname)"
}

# Validate the KV_OFFLOAD tier and (when active) its backend. Exits on bad input.
_kv_offload_validate() {
    case "${KV_OFFLOAD}" in
        none|cpu) ;;
        *)
            echo "Error: unsupported KV_OFFLOAD='${KV_OFFLOAD}' (expected none|cpu)." >&2
            exit 1
            ;;
    esac
    kv_offload_enabled || return 0
    case "${OFFLOAD_BACKEND}" in
        SimpleCPUOffloadConnector|OffloadingConnector|LMCacheConnectorV1) ;;
        *)
            echo "Error: unsupported OFFLOAD_BACKEND='${OFFLOAD_BACKEND}' (expected SimpleCPUOffloadConnector|OffloadingConnector|LMCacheConnectorV1)." >&2
            exit 1
            ;;
    esac
}

# Echo the kv-transfer-config for `vllm serve`: base JSON unchanged when none, else a
# MultiConnector wrapping [<offload sub-connector>, base].
kv_offload_wrap() {
    local base_json="$1"
    if ! kv_offload_enabled; then
        printf '%s' "$base_json"
        return 0
    fi

    _kv_offload_validate

    OFFLOAD_CPU_BYTES="${OFFLOAD_CPU_BYTES}" _BASE_JSON="${base_json}" \
    OFFLOAD_BACKEND="${OFFLOAD_BACKEND}" python3 - <<'PY'
import json, os
base = json.loads(os.environ["_BASE_JSON"])
# vLLM's MultiConnector rebuilds each sub-connector as
# KVTransferConfig(**sub_dict, engine_id=engine_id) (multi_connector.py
# _get_connector_classes_and_configs). It reads engine_id via dict.get() WITHOUT
# popping it, so a sub-connector dict that still carries an "engine_id" key raises
# "got multiple values for keyword argument 'engine_id'". Lift engine_id off the
# base dict to the outer MultiConnector; vLLM's fallback (ktc.get("engine_id",
# outer.engine_id)) then re-applies it to the base (and offload) sub-connector.
engine_id = base.pop("engine_id", None)
backend = os.environ["OFFLOAD_BACKEND"]
offload_role = "kv_consumer" if base.get("kv_role") == "kv_consumer" else "kv_both"
if backend == "LMCacheConnectorV1":
    # LMCache reads its config from the env (LMCACHE_*); see kv_offload_setup_env.
    # Decode workers (base kv_role=kv_consumer) must not save to LMCache: calling
    # from_gpu there dispatches an async store_stream kernel that races with MoRIIO's
    # lazy ibv_reg_mr on the same KV pages. kv_consumer triggers force_skip_save in
    # LMCacheConnectorV1Impl so from_gpu is never called on decode workers.
    offload = {
        "kv_connector": "LMCacheConnectorV1",
        "kv_role": offload_role,
    }
elif backend == "OffloadingConnector":
    offload = {
        "kv_connector": "OffloadingConnector",
        "kv_role": offload_role,
        "kv_connector_extra_config": {
            "cpu_bytes_to_use": int(os.environ["OFFLOAD_CPU_BYTES"]),
        },
    }
else:  # SimpleCPUOffloadConnector
    offload = {
        "kv_connector": "SimpleCPUOffloadConnector",
        "kv_role": offload_role,
        "kv_connector_extra_config": {
            "cpu_bytes_to_use": int(os.environ["OFFLOAD_CPU_BYTES"]),
        },
    }
multi = {
    "kv_connector": "MultiConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "connectors": [offload, base],
    },
}
if engine_id is not None:
    multi["engine_id"] = engine_id
print(json.dumps(multi))
PY
}

# Export env vars the active offload backend reads before `vllm serve`, and create
# the filesystem-tier dir when OFFLOAD_DISK_PATH is set. Submit-time exports win.
kv_offload_setup_env() {
    kv_offload_enabled || return 0

    if [[ "${OFFLOAD_BACKEND}" != "LMCacheConnectorV1" ]]; then
        echo "[kv_offload] ${OFFLOAD_BACKEND} OFFLOAD_CPU_BYTES=${OFFLOAD_CPU_BYTES}"
        return 0
    fi

    export LMCACHE_MAX_LOCAL_CPU_SIZE="${LMCACHE_MAX_LOCAL_CPU_SIZE:-100.0}"
    # Stable hashing across workers so prefix keys match; LMCache prometheus multiproc dir.
    export PYTHONHASHSEED="${PYTHONHASHSEED:-123}"
    export PROMETHEUS_MULTIPROC_DIR="${PROMETHEUS_MULTIPROC_DIR:-/tmp/lmcache_prometheus}"
    mkdir -p "${PROMETHEUS_MULTIPROC_DIR}" || echo "[kv_offload] WARNING: failed to create ${PROMETHEUS_MULTIPROC_DIR}" >&2
    echo "[kv_offload] lmcache CPU tier: LMCACHE_MAX_LOCAL_CPU_SIZE=${LMCACHE_MAX_LOCAL_CPU_SIZE} GB/worker" \
         "PROMETHEUS_MULTIPROC_DIR=${PROMETHEUS_MULTIPROC_DIR}"

    # Optional disk tier: LMCache spills CPU-evicted chunks here (LRU, per-GPU sharded).
    local disk_dir=""
    if [[ -n "${OFFLOAD_DISK_PATH:-}" ]]; then
        disk_dir="$(_kv_offload_fs_dir)"
        mkdir -p "${disk_dir}" || echo "[kv_offload] WARNING: failed to create ${disk_dir}" >&2
        export LMCACHE_LOCAL_DISK="${disk_dir}"
        export LMCACHE_MAX_LOCAL_DISK_SIZE="${LMCACHE_MAX_LOCAL_DISK_SIZE:-0.0}"
        echo "[kv_offload] lmcache disk tier: LMCACHE_LOCAL_DISK=${LMCACHE_LOCAL_DISK}" \
             "LMCACHE_MAX_LOCAL_DISK_SIZE=${LMCACHE_MAX_LOCAL_DISK_SIZE} GB/worker"
    fi
}
