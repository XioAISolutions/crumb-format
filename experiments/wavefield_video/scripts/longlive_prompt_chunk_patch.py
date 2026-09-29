"""Bounded-chunk prompt encoding for pinned LongLive; fail closed.

Why: utils/prompt_conditioning.py upstream encodes ALL block prompts in ONE
text-encoder call, and the T5 attention bias is O(batch) (wan_5b/modules/t5.py
`attn_bias = x.new_zeros(b, n, q, k)`) — 4.25 GiB at 136 block-prompts (180 s
rung) and 7.06 GiB at 226 (300 s rung) against ~1.4-3.4 GiB free: CUDA OOM
before denoise. This patch encodes in bounded chunks (order preserved, cat
dim=0) and keeps the upstream single call whenever chunk >= count.

Env knob: LL_PROMPT_CHUNK (default "16"; 0/negative or >= count -> single
call). Imports stay function-local, matching the idle-offload patch style.

Anchor (must be unique): the flat_prompts / text_encoder call / prompt_embeds
read block in utils/prompt_conditioning.py::encode_prompt_blocks.
"""
import ast
import hashlib
import os
import sys
from pathlib import Path

MARK = "# LL_PROMPT_CHUNK_V1"

_ANCHOR_HEAD = (
    "    flat_prompts = [prompt for sample_prompts in prompt_blocks "
    "for prompt in sample_prompts]\n"
)
ANCHOR = (
    _ANCHOR_HEAD
    + "    conditional_dict = text_encoder(text_prompts=flat_prompts)\n"
    + '    prompt_embeds = conditional_dict["prompt_embeds"]\n'
)

INSERT = '''    # LL_PROMPT_CHUNK_V1: encode block prompts in bounded chunks. The T5
    # attention bias is O(batch) (4.25 GiB @136 prompts, 7.06 GiB @226);
    # chunk >= count keeps the upstream single call unchanged.
    import os as _ll_os
    import torch as _ll_torch

    _ll_chunk = int(_ll_os.environ.get("LL_PROMPT_CHUNK", "16") or "16")
    if _ll_chunk < 1 or _ll_chunk >= len(flat_prompts):
        conditional_dict = text_encoder(text_prompts=flat_prompts)
    else:
        _ll_parts = []
        for _ll_i in range(0, len(flat_prompts), _ll_chunk):
            _ll_part = text_encoder(
                text_prompts=flat_prompts[_ll_i:_ll_i + _ll_chunk]
            )
            _ll_parts.append(_ll_part["prompt_embeds"])
        conditional_dict = {"prompt_embeds": _ll_torch.cat(_ll_parts, dim=0)}
    prompt_embeds = conditional_dict["prompt_embeds"]
'''

REPLACEMENT = _ANCHOR_HEAD + INSERT


def fail(msg):
    raise SystemExit("STRUCTURE FAIL: " + msg)


def _uq(node):
    """unparse with quote style normalized to single quotes (version-stable)."""
    return ast.unparse(node).replace('"', "'")


def _encode_fn(src):
    tree = ast.parse(src)
    fn = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
         and n.name == "encode_prompt_blocks"),
        None,
    )
    if fn is None:
        fail("encode_prompt_blocks() not found")
    return fn


def _assign_idx(stmts, name):
    idxs = []
    for i, st in enumerate(stmts):
        if (isinstance(st, ast.Assign) and len(st.targets) == 1
                and isinstance(st.targets[0], ast.Name)
                and st.targets[0].id == name):
            idxs.append(i)
    return idxs


def assert_structure(src):
    if MARK not in src:
        fail("marker missing")
    fn = _encode_fn(src)
    body = fn.body

    aliases = set()
    imp_idx = []
    for i, st in enumerate(body):
        if isinstance(st, ast.Import):
            imp_idx.append(i)
            aliases.update(a.asname for a in st.names)
    for need in ("_ll_os", "_ll_torch"):
        if need not in aliases:
            fail("missing `import ... as %s`" % need)

    ck = _assign_idx(body, "_ll_chunk")
    if len(ck) != 1:
        fail("expected exactly one _ll_chunk assignment, got %d" % len(ck))
    consts = [n.value for n in ast.walk(body[ck[0]].value)
              if isinstance(n, ast.Constant)]
    if "LL_PROMPT_CHUNK" not in consts:
        fail('_ll_chunk must read env "LL_PROMPT_CHUNK"')
    if "16" not in consts:
        fail('LL_PROMPT_CHUNK default must be "16"')

    gi, gnode = None, None
    for i, st in enumerate(body):
        if isinstance(st, ast.If) and "_ll_chunk" in ast.unparse(st.test):
            if gi is not None:
                fail("duplicate _ll_chunk If")
            gi, gnode = i, st
    if gnode is None:
        fail("_ll_chunk guard If not found")
    if not gnode.orelse:
        fail("guard If must have an else (chunked branch)")

    if not any(
        isinstance(s, ast.Assign) and len(s.targets) == 1
        and isinstance(s.targets[0], ast.Name)
        and s.targets[0].id == "conditional_dict"
        and ast.unparse(s.value) == "text_encoder(text_prompts=flat_prompts)"
        for s in gnode.body
    ):
        fail("single-call branch must assign conditional_dict = "
             "text_encoder(text_prompts=flat_prompts)")

    for_ = next((s for s in gnode.orelse if isinstance(s, ast.For)), None)
    if for_ is None:
        fail("chunked branch must contain a For loop")
    if ast.unparse(for_.iter) != "range(0, len(flat_prompts), _ll_chunk)":
        fail("loop must iterate range(0, len(flat_prompts), _ll_chunk), got: "
             + ast.unparse(for_.iter))
    part_assigns = [
        s for s in for_.body
        if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name)
        and s.targets[0].id == "_ll_part"
    ]
    if len(part_assigns) != 1:
        fail("loop body must assign _ll_part exactly once")
    pu = ast.unparse(part_assigns[0].value)
    if "text_prompts=flat_prompts[_ll_i:_ll_i + _ll_chunk]" not in pu:
        fail("loop call must slice flat_prompts[_ll_i:_ll_i + _ll_chunk], "
             "got: " + pu)
    if not any(
        isinstance(s, ast.Expr)
        and "_ll_parts.append(_ll_part['prompt_embeds'])" in _uq(s)
        for s in for_.body
    ):
        fail('loop body must append _ll_part["prompt_embeds"] in order')
    if not any(
        isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name)
        and s.targets[0].id == "conditional_dict"
        and _uq(s.value)
        == "{'prompt_embeds': _ll_torch.cat(_ll_parts, dim=0)}"
        for s in gnode.orelse
    ):
        fail("chunked branch must set conditional_dict via "
             '{"prompt_embeds": _ll_torch.cat(_ll_parts, dim=0)}')

    flat_idx = _assign_idx(body, "flat_prompts")
    if len(flat_idx) != 1:
        fail("expected exactly one flat_prompts assignment")
    reads = [
        i for i, st in enumerate(body)
        if isinstance(st, ast.Assign) and isinstance(st.targets[0], ast.Name)
        and st.targets[0].id == "prompt_embeds"
        and "conditional_dict['prompt_embeds']" in _uq(st.value)
    ]
    if not reads:
        fail('prompt_embeds must be read from conditional_dict["prompt_embeds"]')
    if not (flat_idx[0] < min(imp_idx) < gi < min(reads)):
        fail("ordering: flat_prompts < imports < guard < prompt_embeds read")


def main():
    src_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/root/LongLive/utils/prompt_conditioning.py")
    src = src_path.read_text()
    if MARK in src:
        assert_structure(src)  # validate, never trust the marker alone
        print("ALREADY PATCHED (structure verified):", src_path)
        return 0
    n = src.count(ANCHOR)
    if n != 1:
        print("ANCHOR FAIL: found %d in %s" % (n, src_path))
        return 2
    new = src.replace(ANCHOR, REPLACEMENT)
    assert_structure(new)
    ast.parse(new)
    backup = Path(str(src_path) + ".before_pchunk")
    if not backup.exists():
        backup.write_text(src)
    tmp = Path(str(src_path) + ".tmp")
    tmp.write_text(new)
    os.replace(tmp, src_path)
    print("APPLIED", hashlib.sha256(new.encode()).hexdigest(), src_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
