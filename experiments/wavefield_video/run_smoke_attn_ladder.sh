#!/usr/bin/env bash
# Portable smoke for the attention-lane suite (card t_d77fd4cd): bash syntax +
# runner-contract tests + report-script behavior. CPU-only, no torch needed.
# The box queue job (gpuq_job_500) additionally runs a micro GPU train and
# asserts the new recipe keys land in the result JSON.
set -euo pipefail
cd "$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY_TEST=${PY_TEST:-python3}

echo '### (1) bash -n on the new runners'
bash -n run_attn_ladder.sh
bash -n run_attn_hybrid_retest.sh

echo '### (2) runner contract tests'
"$PY_TEST" test_attn_ladder_runner.py

echo '### (3) report script on an empty root (must not crash; guard WAIT=2)'
TMPD=$(mktemp -d)
trap 'rm -rf "$TMPD"' EXIT
"$PY_TEST" attn_ladder_report.py --root "$TMPD"
rc=0
"$PY_TEST" attn_ladder_report.py --root "$TMPD" --d5-guard || rc=$?
if [ "$rc" != "2" ]; then
    echo "d5 guard on empty data: expected WAIT(2), got $rc" >&2
    exit 1
fi

echo 'smoke attn-ladder PASS'
