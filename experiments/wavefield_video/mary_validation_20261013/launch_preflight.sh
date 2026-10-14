#!/usr/bin/env bash
set -euo pipefail
cd "$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${PY:?Set PY to the already provisioned Python interpreter}"
: "${OUT:?Set OUT to a fresh receipt directory}"
: "${VAE_DIR:?Existing approved production VAE component directory required}"
: "${VAE_MANIFEST:?Existing publisher snapshot provenance manifest required}"
: "${VAE_MANIFEST_SHA256:?Independently pinned provenance SHA256 required}"
: "${FEATURE_DIR:?Existing frozen DINOv2 snapshot directory required}"
: "${FEATURE_MANIFEST:?Existing DINOv2 provenance manifest required}"
: "${FEATURE_MANIFEST_SHA256:?Independently pinned feature provenance SHA256 required}"
: "${PROBE_VIDEO:?Owned native-resolution video required}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
command -v timeout >/dev/null || { echo "Existing hard-timeout command required" >&2; exit 2; }
exec timeout --kill-after=30s 3600s "$PY" -B run_preflight.py --out "$OUT" --device cuda \
  --backend "${VAE_BACKEND:-wan}" --vae-dir "$VAE_DIR" --vae-manifest "$VAE_MANIFEST" \
  --vae-manifest-sha256 "$VAE_MANIFEST_SHA256" --feature-dir "$FEATURE_DIR" \
  --feature-manifest "$FEATURE_MANIFEST" --feature-manifest-sha256 "$FEATURE_MANIFEST_SHA256" \
  --probe-video "$PROBE_VIDEO"
