"""R14 occlusion-scenario correctness; run directly, no pytest dependency.

Verifies the decisive-experiment data generator: the target is hidden EXACTLY on
the occlusion window and visible otherwise, the scene is deterministic/seedable,
the target's physics continue (and bounce) behind the occluder, and the emergence
exit direction is fully determined by the pre-occlusion state."""
import torch

import data_occlusion as occ


def _target_center_pixel(frames, meta, t):
    """RGB at the target's (rounded) center pixel on frame ``t`` for every seed."""
    pos = meta["pos"][:, t, meta["target_idx"]]                 # [B,2] (x,y)
    x = pos[:, 0].round().long().clamp(0, frames.shape[-1] - 1)
    y = pos[:, 1].round().long().clamp(0, frames.shape[-2] - 1)
    b = torch.arange(frames.shape[0])
    return frames[b, t, :, y, x]                                # [B,3]


def test_hidden_exactly_on_window():
    T = 40
    frames, meta = occ.make_clip_batch(4, T, 64, 64, seed=1, occ_start=10, occ_end=30,
                                       return_meta=True)
    assert frames.shape == (4, T + 1, 3, 64, 64)
    assert meta["occ_start"] == 10 and meta["occ_end"] == 30
    vis = meta["visible"]
    assert vis.shape == (T + 1,)
    for t in range(T + 1):
        px = _target_center_pixel(frames, meta, t)              # [B,3]
        # During the window the occluder is drawn OVER everything in its rect, so
        # the target's center pixel is EXACTLY the occluder color; outside, the pure
        # red target dominates its own center (R stays high regardless of overlaps).
        hidden = (px - occ.OCCLUDER_RGB).abs().amax(-1) < 1e-4
        red_present = px[:, 0] > 0.5
        if 10 <= t < 30:
            assert bool(vis[t]) is False
            assert hidden.all(), f"frame {t}: target must be fully occluded"
        else:
            assert bool(vis[t]) is True
            assert red_present.all(), f"frame {t}: target must be visible"
    print("OK occlusion: target hidden exactly on [occ_start, occ_end), visible otherwise")


def test_deterministic_and_seeded():
    kw = dict(seed=7, occ_start=8, occ_end=24, collisions=True, return_meta=True,
              return_moving=True)
    a = occ.make_clip_batch(3, 30, 64, 64, **kw)
    b = occ.make_clip_batch(3, 30, 64, 64, **kw)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1]["pos"], b[1]["pos"])
    assert torch.equal(a[2], b[2])                              # moving mask reproducible
    c = occ.make_clip_batch(3, 30, 64, 64, **{**kw, "seed": 8})
    assert not torch.equal(a[0], c[0]), "different seed must change the scene"
    g0 = torch.random.get_rng_state().clone()
    occ.make_clip_batch(2, 12, 64, 64, seed=99, return_meta=True)
    assert torch.equal(g0, torch.random.get_rng_state()), "must not consume global RNG"
    print("OK occlusion: deterministic, seedable, and does not touch the global RNG")


def test_target_confined_and_physics_continue():
    # The target stays inside its chamber (so it is always coverable), and its
    # trajectory keeps moving + bouncing through the hidden window.
    T = 80
    frames, meta = occ.make_clip_batch(6, T, 64, 64, seed=3, speed=2.0,
                                       occ_start=20, occ_end=60, return_meta=True)
    tp = meta["pos"][:, :, meta["target_idx"]]                  # [B,T+1,2]
    x0, y0, x1, y1 = [f * 63 for f in occ.CHAMBER_FRAC]
    assert (tp[..., 0] >= x0 - 1e-3).all() and (tp[..., 0] <= x1 + 1e-3).all()
    assert (tp[..., 1] >= y0 - 1e-3).all() and (tp[..., 1] <= y1 + 1e-3).all()
    step = (tp[:, 1:] - tp[:, :-1]).norm(dim=-1)                # per-frame displacement
    hidden = step[:, 20:59]                                     # motion during occlusion
    assert (hidden.mean(dim=1) > 0.1).all(), "target must keep moving while hidden"
    # A bounce shows up as a velocity-sign flip on some seed/axis during the window.
    vel = meta["vel"][:, :, meta["target_idx"]]                 # [B,T+1,2]
    flips = (torch.sign(vel[:, 21:60]) != torch.sign(vel[:, 20:59])).any()
    assert bool(flips), "target should bounce off a chamber wall during occlusion"
    print("OK occlusion: target confined to chamber, keeps moving and bouncing while hidden")


def test_exit_direction_determined_by_pre_occlusion_state():
    # Same seed => identical emergence direction (determinism of the exit).
    T, os_, oe = 60, 12, 40
    f1, m1 = occ.make_clip_batch(5, T, 64, 64, seed=11, occ_start=os_, occ_end=oe,
                                 return_meta=True)
    f2, m2 = occ.make_clip_batch(5, T, 64, 64, seed=11, occ_start=os_, occ_end=oe,
                                 return_meta=True)
    v1 = m1["vel"][:, oe, m1["target_idx"]]
    v2 = m2["vel"][:, oe, m2["target_idx"]]
    assert torch.equal(torch.sign(v1), torch.sign(v2))
    # The emergence velocity is a deterministic function of the initial velocity and
    # elapsed bounces -- non-degenerate across seeds (both exit signs appear).
    dirs = torch.sign(occ.make_clip_batch(64, T, 64, 64, seed=5, occ_start=os_,
                                          occ_end=oe, return_meta=True)[1]["vel"]
                      [:, oe, occ.TARGET_IDX])
    assert dirs[:, 0].unique().numel() == 2, "exit x-direction must vary across seeds"
    print("OK occlusion: emergence exit direction is deterministic and pre-occlusion-determined")


if __name__ == "__main__":
    torch.set_num_threads(1)
    test_hidden_exactly_on_window()
    test_deterministic_and_seeded()
    test_target_confined_and_physics_continue()
    test_exit_direction_determined_by_pre_occlusion_state()
    print("ALL occlusion data tests passed")
