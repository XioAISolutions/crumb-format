#!/usr/bin/env python3
"""LL I2V support: longlive_long.py gains --i2v (+ image-dir counting, overlay
flags); run_longlive.sh gains I2V=1 (i2v base config, dir prompts, --i2v).
Fail-closed on any anchor mismatch."""
import ast
import pathlib
import shutil
import sys


def count_else(src):
    return sum(1 for n in ast.walk(ast.parse(src))
               if isinstance(n, ast.If) and n.orelse)


def patch_file(path, subs):
    src = orig = path.read_text()
    for i, (old, new) in enumerate(subs, 1):
        n = src.count(old)
        if n != 1:
            sys.exit(f"ANCHOR {i}: count={n} (want 1) in {path}:\n  {old[:110]!r}")
        src = src.replace(old, new)
    if src == orig:
        sys.exit(f"no change in {path}")
    bak = path.with_suffix(path.suffix + ".before_i2v")
    shutil.copy2(path, bak)
    path.write_text(src)
    print(f"PATCHED {path} (backup {bak.name})")


E = pathlib.Path("/workspace/slava/exp/pr63")
LLF = E / "longlive_long.py"
RUN = E / "run_longlive.sh"

ll_subs = [
    ("                  seed=0, fp8=True, compile=False, relative_rope=True):",
     "                  seed=0, fp8=True, compile=False, relative_rope=True, i2v=False):"),
    ('    if fp8:\n        cfg.pop("model_quant", None)\n    return cfg',
     '    if fp8:\n        cfg.pop("model_quant", None)\n'
     '    if i2v:\n        cfg["i2v"] = True\n'
     '        cfg.setdefault("algorithm", {})["i2v"] = True\n'
     '        cfg.setdefault("inference", {})["independent_first_frame"] = True\n'
     '        cfg.pop("adapter", None)\n'
     '        cfg.setdefault("checkpoints", {}).pop("lora_ckpt", None)\n'
     '    return cfg'),
    ('    cap = prompts / "caption" if (prompts / "caption").is_dir() else prompts\n'
     '    return sum(1 for d in cap.iterdir() if d.is_dir())',
     '    img_dir = prompts / "images" if (prompts / "images").is_dir() else prompts\n'
     '    exts = {".png", ".jpg", ".jpeg", ".webp"}\n'
     '    imgs = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]\n'
     '    if imgs:\n        return len(imgs)\n'
     '    cap = prompts / "caption" if (prompts / "caption").is_dir() else prompts\n'
     '    return sum(1 for d in cap.iterdir() if d.is_dir())'),
    ('                        relative_rope=(a.relative_rope == "on"))',
     '                        relative_rope=(a.relative_rope == "on"), i2v=a.i2v)'),
    ('            s.add_argument("--relative-rope", choices=["on", "off"], default="on",',
     '            s.add_argument("--i2v", action="store_true",\n'
     '                           help="i2v: --prompts is an images/ data dir")\n'
     '            s.add_argument("--relative-rope", choices=["on", "off"], default="on",'),
]
run_subs = [
    ('base_args=(); [ -n "$BASE_CONFIG" ] && base_args=(--base-config "$BASE_CONFIG")',
     'I2V=${I2V:-0}\n'
     '[ "$I2V" = "1" ] && [ -z "$BASE_CONFIG" ] && BASE_CONFIG="$LL/configs/inference_i2v.yaml"\n'
     'base_args=(); [ -n "$BASE_CONFIG" ] && base_args=(--base-config "$BASE_CONFIG")\n'
     'i2v_args=(); [ "$I2V" = "1" ] && i2v_args=(--i2v)'),
    ('SEED=$SEED ENCODER=$ENCODER BASE_CONFIG=$BASE_CONFIG"',
     'SEED=$SEED ENCODER=$ENCODER BASE_CONFIG=$BASE_CONFIG I2V=$I2V"'),
    ('[ -f "$PROMPTS" ] || { echo "no prompts file $PROMPTS" >&2; exit 2; }',
     'if [ "$I2V" = "1" ]; then\n'
     '    [ -d "$PROMPTS" ] || { echo "no i2v data dir $PROMPTS" >&2; exit 2; }\n'
     'else\n'
     '    [ -f "$PROMPTS" ] || { echo "no prompts file $PROMPTS" >&2; exit 2; }\n'
     'fi'),
    ('--relative-rope "$RELATIVE_ROPE" ${base_args[@]+"${base_args[@]}"}',
     '--relative-rope "$RELATIVE_ROPE" ${i2v_args[@]+"${i2v_args[@]}"} ${base_args[@]+"${base_args[@]}"}'),
]

n0 = count_else(LLF.read_text())
patch_file(LLF, ll_subs)
patch_file(RUN, run_subs)
n1 = count_else(LLF.read_text())
assert n1 == n0, f"else-attachment changed: {n0} -> {n1}"
print(f"AST OK: else-if count unchanged ({n0}); now run bash -n + pytest")
