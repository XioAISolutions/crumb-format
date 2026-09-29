"""Guard unused negative caches in pinned LongLive; fail closed on source drift."""
import ast
from pathlib import Path
import sys


def patched_source(source):
    marker = '# LONGLIVE_CFG_CACHE_GUARD_V1'
    if marker in source:
        raise ValueError('already patched; do not apply twice')
    tree = ast.parse(source)
    edits = []
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef) or cls.name != 'CausalDiffusionInferencePipeline':
            continue
        for method in cls.body:
            if not isinstance(method, ast.FunctionDef) or method.name not in ('_initialize_kv_cache', '_initialize_crossattn_cache'):
                continue
            for node in ast.walk(method):
                if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                    continue
                fun = node.value.func
                if isinstance(fun, ast.Attribute) and fun.attr == 'append' and isinstance(fun.value, ast.Name) and fun.value.id in ('kv_cache_neg', 'crossattn_cache_neg'):
                    edits.append((node.lineno - 1, node.end_lineno, node.col_offset))
    if len(edits) != 3:
        raise ValueError(f'expected exactly 3 negative cache append sites, found {len(edits)}')
    lines = source.splitlines(keepends=True)
    for start, end, indent in sorted(edits, reverse=True):
        body = ['    ' + line for line in lines[start:end]]
        lines[start:end] = [' ' * indent + 'if self.guidance_scale != 1.0:  ' + marker + '\n'] + body
    result = ''.join(lines)
    anchor = '        if self.kv_cache_pos is None:\n'
    if result.count(anchor) != 1:
        raise ValueError('cache-reinitialization anchor changed')
    result = result.replace(anchor, '        if self.kv_cache_pos is None or bool(self.kv_cache_neg) != use_cfg:\n            self.clear_cache()  # release stale caches before reallocating\n')
    reset = '\n                self.crossattn_cache_neg[block_index]["is_init"] = False\n'
    if result.count(reset) != 1:
        raise ValueError('chunk reset anchor changed')
    result = result.replace(reset, '\n                if use_cfg:\n    ' + reset[1:])
    ast.parse(result)
    return result


if __name__ == '__main__':
    source, target = map(Path, sys.argv[1:])
    text = patched_source(source.read_text())
    try:
        with open(target, 'x') as fh:    # exclusive create: never clobbers another writer
            fh.write(text)
    except FileExistsError:
        raise SystemExit(f'refusing to overwrite {target}')
    print(f'wrote {target}')
