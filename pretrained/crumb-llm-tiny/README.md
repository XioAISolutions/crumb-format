# crumb-llm-tiny

Tiny local pretrained CrumbLLM checkpoint for registry and smoke testing.

This is a small character-tokenizer Wave-Field LM trained on the bundled
crumb-format examples corpus. It is meant to prove the standalone model
registry, download, load, serve, and generation path. It is not a strong
general-purpose language model.

## Training

- Config: `crumb_llm/configs/tiny.json`
- Steps: 200
- Seed: 0
- Parameters: 230,916
- Final loss: 1.6302
- Final BPC: 2.3518
- Checkpoint: `ckpt.pt`

## Use

```bash
crumb-llm register crumb-llm-tiny-local --path pretrained/crumb-llm-tiny --alias local-tiny
crumb-llm download crumb-llm-tiny-local
crumb-llm generate --ckpt ~/.cache/crumb-llm/models/crumb-llm-tiny-local --prompt "BEGIN CRUMB"
```

## Publish Prep

```bash
python scripts/publish_crumb_llm_model.py
```

That command writes a HuggingFace-compatible folder under
`dist/crumb-llm-tiny-hub/`, plus a registry JSON containing file checksums.
Add `--upload` only after `huggingface_hub` auth is configured.
