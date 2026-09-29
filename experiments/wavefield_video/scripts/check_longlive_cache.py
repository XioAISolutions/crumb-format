"""CPU regression on exact LongLive source; no weights/GPU/import side effects."""
import ast
import os
from pathlib import Path
from typing import List, Optional
import torch

SOURCE = Path(os.environ['LL_PIPELINE_SOURCE'])

def pipeline(scale):
    tree = ast.parse(SOURCE.read_text())
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CausalDiffusionInferencePipeline')
    names = {'_initialize_kv_cache', '_initialize_crossattn_cache', 'clear_cache', 'inference'}
    methods = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in names]
    inference = next(n for n in methods if n.name == 'inference')
    stop = next(i for i, n in enumerate(inference.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'current_start_frame' for t in n.targets))
    inference.body = inference.body[:stop] + [ast.Return(value=ast.Name(id='output', ctx=ast.Load()))]
    cls = ast.ClassDef(name='CacheOnly', bases=[], keywords=[], body=methods, decorator_list=[])
    mod = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    ns = dict(torch=torch, List=List, Optional=Optional, wan_default_config={'tiny': {'num_heads': 2, 'head_dim': 4}}, encode_prompt_blocks=lambda *args: ({}, []))
    exec(compile(mod, str(SOURCE), 'exec'), ns)
    p = ns['CacheOnly']()
    p.model_name = 'tiny'
    p.num_transformer_blocks = 2
    p.local_attn_size = 4
    p.frame_seq_length = 4
    p.num_frame_per_block = 2
    p.quantize_kv = False
    p.guidance_scale = scale
    p.negative_prompt = ''
    p.text_encoder = lambda **kwargs: {}
    p.independent_first_frame = False
    p.clear_cache()
    return p

def allocate(p):
    return p.inference(torch.zeros(1, 4, 2, 4, 4), [['prompt']], return_latents=True)

def test_no_cfg_has_no_negative_allocations():
    p = pipeline(1.0)
    allocate(p)
    assert p.kv_cache_neg == [], 'unused negative KV allocated with guidance disabled'
    assert p.crossattn_cache_neg == [], 'unused negative cross-attention allocated'
    assert len(p.kv_cache_pos) == p.num_transformer_blocks


def test_cfg_toggle_allocates_only_required_caches():
    p = pipeline(1.0)
    allocate(p)
    p.guidance_scale = 5.0
    allocate(p)
    assert len(p.kv_cache_neg) == p.num_transformer_blocks
    assert len(p.crossattn_cache_neg) == p.num_transformer_blocks
    p.guidance_scale = 1.0
    allocate(p)
    assert p.kv_cache_neg == [] and p.crossattn_cache_neg == []


def test_cfg_positive_cache_is_unchanged():
    a, b = pipeline(1.0), pipeline(5.0)
    allocate(a)
    allocate(b)
    assert len(b.kv_cache_neg) == b.num_transformer_blocks
    for cache_name in ('kv_cache_pos', 'crossattn_cache_pos'):
        for x, y in zip(getattr(a, cache_name), getattr(b, cache_name)):
            assert x.keys() == y.keys()
            for key in x:
                if torch.is_tensor(x[key]):
                    assert torch.equal(x[key], y[key])
                else:
                    assert x[key] == y[key]

if __name__ == '__main__':
    test_no_cfg_has_no_negative_allocations()
    test_cfg_toggle_allocates_only_required_caches()
    test_cfg_positive_cache_is_unchanged()
    print('3 CPU cache regression checks passed')

