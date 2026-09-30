"""longlive_long: I2V additions — image data-dir counting and overlay flags."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import longlive_long as L  # noqa: E402

BASE = {
    "model_kwargs": {"model_name": "Wan2.2-TI2V-5B", "num_frame_per_block": 8, "local_attn_size": 32},
    "num_output_frames": 8,
    "data": {"data_path": "example/long_example.txt", "image_or_video_shape": [1, 8, 48, 44, 80]},
    "inference": {"sampling_steps": 4, "sink_size": 8, "streaming_vae": True},
    "checkpoints": {"generator_ckpt": "/path/to/model_bf16.pt", "lora_ckpt": "/p/lora.pt"},
    "adapter": {"type": "lora"},
    "fp8_quant": True,
    "logging": {"seed": 0},
}


def test_n_prompts_of_image_layouts(tmp_path):
    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "a.png").write_bytes(b"x")
    (flat / "a.txt").write_text("cap\n")
    (flat / "b.jpg").write_bytes(b"y")
    (flat / "b.txt").write_text("cap2\n")
    assert L.n_prompts_of(flat) == 2
    nested = tmp_path / "nested"
    (nested / "images").mkdir(parents=True)
    (nested / "images" / "c.png").write_bytes(b"z")
    assert L.n_prompts_of(nested) == 1
    txt = tmp_path / "p.txt"
    txt.write_text("one\n\n two \n")
    assert L.n_prompts_of(txt) == 2


def test_overlay_i2v_flags(tmp_path):
    cfg = L.build_overlay(BASE, latent_frames=8, prompts="p", ckpt="c", out_dir=tmp_path,
                          window=24, sink=4, i2v=True)
    assert cfg["i2v"] is True
    assert cfg["algorithm"]["i2v"] is True
    assert cfg["inference"]["independent_first_frame"] is True
    assert "adapter" not in cfg
    assert "lora_ckpt" not in cfg["checkpoints"]
    cfg2 = L.build_overlay(BASE, latent_frames=8, prompts="p", ckpt="c", out_dir=tmp_path)
    assert "i2v" not in cfg2
