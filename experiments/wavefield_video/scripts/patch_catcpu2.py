import ast
import re

p = "/root/opsd-v/wan/modules/vae.py"
src = open(p).read()
lines = src.split("\n")
d = [i for i, l in enumerate(lines) if l.strip().startswith("def decode(")]
assert len(d) >= 1, ("defs", d)
start = d[0]  # the plain decode (traceback call site)
target = None
for i in range(start + 1, start + 90):
    if re.match(r"^ {0,4}def ", lines[i]):
        break
    if "torch.cat([out, out_], 2)" in lines[i]:
        target = i
        break
assert target is not None, "cat not found in decode()"
lines[target] = lines[target].replace(
    "torch.cat([out, out_], 2)",
    "torch.cat([out.cpu(), out_.cpu()], 2)  # [zeph] CPU-accumulate (24 GB card)")
open(p + ".before_catcpu", "w").write(src)
out = "\n".join(lines)
ast.parse(out)
b = sum(1 for n in ast.walk(ast.parse(src)) if isinstance(n, ast.If) and n.orelse)
a = sum(1 for n in ast.walk(ast.parse(out)) if isinstance(n, ast.If) and n.orelse)
assert b == a, ("else drift", b, a)
open(p, "w").write(out)
print("PATCHED3b line", target + 1, "else-count", a)
