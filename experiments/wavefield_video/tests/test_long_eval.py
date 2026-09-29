"""long_eval: synthetic diagnostic flags and explicit limits on quality claims."""
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


def test_stable_has_no_flags_but_visual_quality_remains_unverified():
    r = run("stable")
    assert r["verdict"] == "NO_FLAGS", r["windows"]
    assert r["semantic_quality"] == "unverified"
    assert r["independent_pixels_review_required"] is True
    assert r["reference"] == "window_0_unverified"
    assert r["drift_reference"] == "window_0_self_similarity"
    assert "pixel thumbnails" in r["diagnostic_scope"]
    assert "heuristic" in r["diagnostic_scope"]
    assert any("coherent_horizon_s" in limit and "alias" in limit for limit in r["limitations"])
    assert r["diagnostic_horizon_s"] == r["coherent_horizon_s"]
    assert r["coherent_horizon_s"] == r["duration_s"]


@pytest.mark.parametrize("kind,mode", [("fade", "fade"), ("freeze", "freeze"),
                                       ("drift", "drift"), ("flatten", "flatten")])
def test_failure_modes(kind, mode):
    r = run(kind)
    assert r["verdict"].startswith("FAIL"), r["windows"]
    assert any(mode in w["fails"] for w in r["windows"]), [w["fails"] for w in r["windows"]]
    assert 30 <= r["coherent_horizon_s"] <= 90
    assert not r["windows"][0]["fails"]


def test_reads_mp4_stream(tmp_path, capsys):
    imageio = pytest.importorskip("imageio.v2")
    pytest.importorskip("imageio_ffmpeg")
    p = tmp_path / "v.mp4"
    w = imageio.get_writer(str(p), fps=8, codec="libx264", macro_block_size=1, ffmpeg_log_level="error")
    for i in range(8 * 70):
        w.append_data(scene(i / 8, "stable"))
    w.close()
    rep = E.main([str(p), "--encoder", "pixel", "--out", str(tmp_path / "r.json")])
    output = capsys.readouterr().out
    assert "PASS" not in output and "coherent horizon" not in output
    assert "semantic quality unverified" in output
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
        output = capsys.readouterr().out
        assert "PASS" not in output and "coherent horizon" not in output
        assert "semantic quality unverified" in output
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
    assert ok["baseline"]["sat"] < 1e-6 and ok["verdict"] == "NO_FLAGS", ok["windows"]
    bad = E.evaluate(_flat_samples((1.0, 0.3, 0.3)), E.PixelEncoder(), window=30)
    assert any("colour" in w["fails"] for w in bad["windows"]), bad["windows"]


def test_negative_similarity_flags_drift():
    """Anti-correlated features must flag drift against the first window."""
    class Flip(E.PixelEncoder):
        def __call__(self, frames):
            out = super().__call__(frames)
            self.t = getattr(self, "t", 0) + len(frames)
            return out if self.t <= 60 else -out          # after 30 s: anti-correlated features
    r = E.evaluate(samples("stable"), Flip(), window=30)
    assert r["windows"][1]["sim_to_first"] <= 0
    assert all("drift" in w["fails"] for w in r["windows"][1:]), [w["fails"] for w in r["windows"]]
    assert r["verdict"].startswith("FAIL")


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


def test_early_positive_drift_does_not_define_its_own_baseline():
    """Already degraded window 1 must not normalize itself (and later damage) to 1."""
    class EarlyDrift:
        name = identity = "synthetic"

        def __init__(self):
            self.offset = 0

        def __call__(self, frames):
            indices = np.arange(self.offset, self.offset + len(frames))
            self.offset += len(frames)
            cosine = np.where(indices < 60, 1.0, 0.2)
            return np.stack((cosine, np.sqrt(1 - cosine ** 2)), axis=1)

    r = E.evaluate(samples("stable"), EarlyDrift(), window=30)
    assert r["baseline"]["self_sim"] == pytest.approx(1.0)
    assert r["windows"][1]["sim_to_first"] == pytest.approx(0.2)
    assert all("drift" in w["fails"] for w in r["windows"][1:])
    assert r["windows"][1]["drift_ratio"] == pytest.approx(0.2)
    assert r["coherent_horizon_s"] == 30


def test_auto_fallback_reports_pixel_scope_without_dino_evidence(monkeypatch):
    def unavailable():
        raise ImportError("test: DINO weights unavailable")

    monkeypatch.setattr(E, "Dinov2Encoder", unavailable)
    r = E.evaluate(samples("stable"), E.make_encoder("auto"), window=30)
    assert r["encoder"] == "pixel"
    assert "pixel thumbnails" in r["diagnostic_scope"]
    assert "DINO" not in r["diagnostic_scope"]
    assert r["semantic_quality"] == "unverified"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("source", ["pixels", "features"])
def test_nonfinite_data_cannot_report_no_flags_even_in_window_zero(source, value):
    class Encoder:
        name = identity = "synthetic"

        def __call__(self, frames):
            return np.tile([1.0, value if source == "features" else 0.0], (len(frames), 1))

    frame = scene(0, "stable").astype(np.float32)
    if source == "pixels":
        frame[0, 0, 0] = value
    with pytest.raises(ValueError, match="non-finite"):
        E.evaluate([(0, frame)], Encoder(), window=30)


def test_initial_static_fine_texture_is_an_explicitly_unverified_miss():
    """A bad opening can still be statistically stable: no semantic certification."""
    texture = np.random.default_rng(42).integers(50, 206, (H, W, 3), dtype=np.uint8)
    r = E.evaluate(((i / 2, texture) for i in range(180)), E.PixelEncoder(), window=30)
    assert r["verdict"] == "NO_FLAGS"
    assert all(not row["fails"] for row in r["windows"])
    assert r["reference"] == "window_0_unverified"
    assert r["semantic_quality"] == "unverified"
    assert r["independent_pixels_review_required"] is True
