import ast

p = "/root/opsd-v/pipeline/causal_inference_lmdb.py"
src = open(p).read()
old = '        video = (video * 0.5 + 0.5).clamp(0, 1)\n'
new = ('        video = video.cpu()  # [zeph 2026-09-30] offload before clamp (24 GB card)\n'
       + old)
assert src.count(old) == 1, ("anchor", src.count(old))
open(p + ".before_vidcpu", "w").write(src)
out = src.replace(old, new)
ast.parse(out)
b = sum(1 for n in ast.walk(ast.parse(src)) if isinstance(n, ast.If) and n.orelse)
a = sum(1 for n in ast.walk(ast.parse(out)) if isinstance(n, ast.If) and n.orelse)
assert b == a, ("else drift", b, a)
open(p, "w").write(out)
print("PATCHED2, else-count", a)
