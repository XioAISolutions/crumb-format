# Receipt: W16 eval-batch-1 fix (2026-09-30 pump)

## What happened
- Slice copies 988-993 were consumed in ~5 min (16:56-17:00Z): each resumed _W_soft_s1 at step 4000/4000 (training complete) and died on the FINAL EVAL -- CUDA OOM at the auto-batch floor (eb 64->32->16->8; "Tried to allocate 8.00 GiB", ~22.33 GiB in-use). Same failure class as the W_half_seq_cw_s0 seq-arm eval fixed in the pr63 tree at 12:14 (eb=1 passes, ~24 s).
- The fix had not reached the tree the suite runs from (exp/wavefield_video): its runner had no eval-batch anywhere, train_compare.py hardcoded eb=64.

## Fix (this dir carries the exact files)
- run_long_horizon.sh: `--eval-batch 1` on BOTH eval paths (compare BASE + train_long); md5 51ebde29 -> 0fdd4a92. Box backup .bak_pre_evalbatchpump_20260930.
- train_compare.py: new `--eval-batch` arg (default 64 = old hardcoded value), `eb = a.eval_batch`; md5 5e235d4c -> a5795254.
- Verified: md5 read-back (box == local), bash -n, py_compile, live parse test rc=0 (`--eval-batch 1` coexists with `--eval-batches`).

## Queue
- gpuq copies 994-999 staged in gpu_queue/pending (byte-identical body to copy 993; headers bumped). First to fire closes _W_soft_s1's eval, then s1 arms continue. Fire order: c01af 1-4 + studio ea8 (pre-fire checked) -> 994-999.
- c01af + studio jobs pre-fire check: PY gate, PYTORCH_CUDA_ALLOC_CONF, own log, real timeout -- all present. ea8 prompts file staged (matches proven ac2 shape).

## Flags for review
- Slice v2.1 OOM-rescue is a dead path (status tag carries a leading `_` -> looks for log__<tag>.txt). Left as-is on purpose: fail-loud beats silent retry loops for eval OOMs; matches "real failures stop".
- queue_jobs/long_horizon_2026-09-28/tpl_long_horizon_slice.sh (master) is stale (pre-v2.1/v2.2) -- copy new slices from 994-999, not it.

## Proof boundary
- Parse/syntax/md5 verified; GPU proof = copy 994 completing _W_soft_s1's eval (queued, ~2h behind the seq slice at write time).
