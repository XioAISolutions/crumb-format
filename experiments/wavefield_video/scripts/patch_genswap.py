import ast

p = "/root/opsd-v/pipeline/causal_inference_lmdb.py"
src = open(p).read()
old = '        video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)\n'
new = ('        # [zeph 2026-09-30] 24 GB card: free the generator before VAE decode\n'
       '        try:\n'
       '            self.generator.to("cpu")\n'
       '            if torch.cuda.is_available():\n'
       '                torch.cuda.empty_cache()\n'
       '        except Exception as _e:\n'
       '            print("[zeph] generator offload skipped:", _e)\n'
       + old)
assert src.count(old) == 1, ("anchor", src.count(old))
open(p + ".before_genswap", "w").write(src)
out = src.replace(old, new)
ast.parse(out)
b = sum(1 for n in ast.walk(ast.parse(src)) if isinstance(n, ast.If) and n.orelse)
a = sum(1 for n in ast.walk(ast.parse(out)) if isinstance(n, ast.If) and n.orelse)
assert b == a, ("else drift", b, a)
open(p, "w").write(out)
print("PATCHED, else-count", a)
