#!/bin/bash
# D5 (g64) CONDITIONAL slice (card t_d77fd4cd): fires only if the pre-registered
# guard says TRIGGER -- attention competitive (cr >= 0.5, cr >= 0.9x wave,
# mse/ctl <= 1.15x wave) at >= 1 g32 rung. Guard verdicts: 0 TRIGGER -> run the
# D5 pair (own OUT dir); 1 SKIP -> record once, never run D5; 2 WAIT ->
# insufficient g32 data, a later copy re-checks. See IMPL_NOTES_ATTN_LADDER.md.
set -euo pipefail
cd /workspace/slava/exp/wavefield_video
PY=/workspace/slava/comfy-house/venv/bin/python
D5=runs_attn_ladder_d5/d5_status.txt
mkdir -p runs_attn_ladder_d5
if [ -f "$D5" ] && grep -q "^SKIPPED" "$D5"; then exit 0; fi
rc=0
$PY attn_ladder_report.py --d5-guard > runs_attn_ladder_d5/d5_guard_last.txt 2>&1 || rc=$?
cat runs_attn_ladder_d5/d5_guard_last.txt
if [ "$rc" = "1" ]; then
    printf 'SKIPPED guard-unmet %s\n' "$(date -u +%FT%TZ)" >> "$D5"
    exit 0
fi
if [ "$rc" = "2" ]; then
    echo "D5 guard: WAIT (insufficient g32 data yet) $(date -u +%FT%TZ)"
    exit 0
fi
if [ "$rc" != "0" ]; then
    echo "D5 guard error rc=$rc" >&2
    exit 1
fi
grep -q "^TRIGGERED" "$D5" 2>/dev/null || printf 'TRIGGERED %s\n' "$(date -u +%FT%TZ)" >> "$D5"
export RESUME=1 CONST_LR=1 STEPS=8000 SLICE_S=4200 RUNGS="D5"
export PY OUT=runs_attn_ladder_d5
exec bash run_attn_ladder.sh
