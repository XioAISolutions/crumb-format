"""long_eval: each pre-registered failure mode fires on a synthetic clip built
to show it, and a stable moving scene passes."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import long_eval as E  # noqa: E402

H, W = 36, 64


def scene(t, kind):
    rng = np.random.default_rng(0)
    bg = (rng.random((H, W, 3)) * 120 + 60).astype(np.float32)          # textured background
    x = int(8 + 40 * (0.5 + 0.5 * np.sin(t * 1.3)))
    f = bg.copy()
    if kind == "freeze" and t > 60:
        x = int(8 + 40 * (0.5 + 0.5 * np.sin(60 * 1.3)))
    f[10:22, x:x + 10] = [230, 40, 40]
    if kind == "fade":
        f *= max(0.0, 1 - max(0.0, t - 45) / 60)
    if kind == "drift" and t > 60:
        f = (np.random.default_rng(1).random((H, W, 3)) * 255).astype(np.float32)
        f[10:22, x:x + 10] = [40, 40, 230]
    if kind == "flatten":
        a = min(1.0, max(0.0, t - 45) / 30)
        f = f * (1 - a) + f.mean() * a
    return np.clip(f, 0, 255).astype(np.uint8)


def samples(kind, dur=120, per_sec=2):
    for i in range(int(dur * per_sec)):
        t = i / per_sec
        yield t, scene(t, kind)


def run(kind):
    return E.evaluate(samples(kind), E.PixelEncoder(), window=30)


def test_stable_passes():
    r = run("stable")
    assert r["verdict"] == "PASS", r["windows"]
    assert r["coherent_horizon_s"] == r["duration_s"]


@pytest.mark.parametrize("kind,mode", [("fade", "fade"), ("freeze", "freeze"),
                                       ("drift", "drift"), ("flatten", "flatten")])
def test_failure_modes(kind, mode):
    r = run(kind)
    assert r["verdict"].startswith("FAIL"), r["windows"]
    assert any(mode in w["fails"] for w in r["windows"]), [w["fails"] for w in r["windows"]]
    assert 30 <= r["coherent_horizon_s"] <= 90
    assert not r["windows"][0]["fails"]


def test_reads_mp4_stream(tmp_path):
    imageio = pytest.importorskip("imageio.v2")
    pytest.importorskip("imageio_ffmpeg")
    p = tmp_path / "v.mp4"
    w = imageio.get_writer(str(p), fps=8, codec="libx264", macro_block_size=1, ffmpeg_log_level="error")
    for i in range(8 * 70):
        w.append_data(scene(i / 8, "stable"))
    w.close()
    rep = E.main([str(p), "--encoder", "pixel", "--out", str(tmp_path / "r.json")])
    assert rep["n_samples"] == 140 and len(rep["windows"]) == 3
    assert (tmp_path / "r.json").exists()
    # --reuse: a matching receipt is kept; another video / argument / evaluator is not
    args = [str(p), "--encoder", "pixel", "--out", str(tmp_path / "r.json"), "--reuse"]
    calls = []
    real = E.evaluate
    E.evaluate = lambda *x, **k: calls.append(1) or real(*x, **k)
    try:
        E.main(args)
        assert calls == []                                         # reused
        E.main(args + ["--window", "20"])
        assert calls == [1]                                        # other arguments: recomputed
        E.main(args + ["--window", "20"])
        assert calls == [1]                                        # and now reused
        with open(p, "ab") as f:
            f.write(b"\0")                                         # same name, other bytes
        E.main(args + ["--window", "20"])
        assert calls == [1, 1]
        r = __import__("json").load(open(tmp_path / "r.json"))
        r["receipt"]["evaluator"] = "older"                        # produced by another evaluator
        __import__("json").dump(r, open(tmp_path / "r.json", "w"))
        E.main(args + ["--window", "20"])
        assert calls == [1, 1, 1]
        # "auto" resolving to another encoder (DINO became available) is a new receipt
        auto = [x if x != "pixel" else "auto" for x in args] + ["--window", "20"]
        real_make = E.make_encoder
        E.make_encoder = lambda name: E.PixelEncoder()           # auto fell back to pixel
        E.main(auto)
        n = len(calls)
        E.main(auto)
        assert len(calls) == n                                     # same effective encoder: reused

        class OtherEncoder(E.PixelEncoder):
            identity = "dinov2:facebook/dinov2-small:0123456789abcdef"
        E.make_encoder = lambda name: OtherEncoder()
        E.main(auto)
        assert len(calls) == n + 1                                 # resolved differently: recomputed
        E.make_encoder = real_make
    finally:
        E.evaluate = real


def _flat_samples(colour_after, dur=90, per_sec=2):
    """Grayscale textured opening with a moving square; after 45 s the scene turns colour_after."""
    rng = np.random.default_rng(0)
    bg = (rng.random((H, W)) * 120 + 60).astype(np.float32)
    for i in range(dur * per_sec):
        t = i / per_sec
        x = int(8 + 40 * (0.5 + 0.5 * np.sin(t * 1.3)))
        g = bg.copy()
        g[10:22, x:x + 10] = 220
        f = np.repeat(g[..., None], 3, -1)
        if t > 45:
            f = f * np.asarray(colour_after, np.float32)
        yield t, np.clip(f, 0, 255).astype(np.uint8)


def test_zero_saturation_baseline_still_bounds_colour():
    ok = E.evaluate(_flat_samples((1.0, 1.0, 1.0)), E.PixelEncoder(), window=30)
    assert ok["baseline"]["sat"] < 1e-6 and ok["verdict"] == "PASS", ok["windows"]
    bad = E.evaluate(_flat_samples((1.0, 0.3, 0.3)), E.PixelEncoder(), window=30)
    assert any("colour" in w["fails"] for w in bad["windows"]), bad["windows"]


def test_black_opening_still_bounds_luma():
    def samples():
        for i in range(180):
            t = i / 2
            v = 0 if t < 30 else 200
            f = np.full((H, W, 3), v, np.uint8)
            f[10:22, int(t) % 50:int(t) % 50 + 10] = 255 if t >= 30 else 3
            yield t, f
    r = E.evaluate(samples(), E.PixelEncoder(), window=30)
    assert r["baseline"]["luma"] < 0.02
    assert any("fade" in w["fails"] for w in r["windows"]), r["windows"]
