"""R12 data regression checks; run directly, no pytest dependency required."""
import torch

from data import (CONTACT, N_BALLS, PALETTE_ANCHORS, RADIUS, SPEED,
                  _resolve_collisions, make_clip_batch, moving_mask)


def _r11_default(bs, T, H, W, seed, *, kicks=False, collisions=False):
    """Frozen pre-R12 three-ball path, evaluated on the current CPU backend.

    Compare tensors byte-for-byte rather than pinning exp() hashes that can
    differ between CPU architectures. This includes the original collision
    loop, random draw order, colors and frame arithmetic.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    nb = 3
    maxs = torch.tensor([W - 1.0, H - 1.0])
    pos = torch.rand(bs, nb, 2, generator=g) * maxs
    vel = (torch.rand(bs, nb, 2, generator=g) * 2 - 1) * 1.15
    anchors = torch.tensor([[1.0, 0.22, 0.22], [0.25, 1.0, 0.30], [0.30, 0.45, 1.0]])
    idx = torch.rand(bs, nb, generator=g).argsort(1)
    col = anchors[idx] * (0.85 + 0.15 * torch.rand(bs, nb, 1, generator=g))
    ys = torch.arange(H).view(1, 1, H, 1).float()
    xs = torch.arange(W).view(1, 1, 1, W).float()
    frames, positions = [], []
    for t in range(T + 1):
        dy = ys - pos[:, :, 1][:, :, None, None]
        dx = xs - pos[:, :, 0][:, :, None, None]
        blobs = torch.exp(-(dy * dy + dx * dx) / (2 * 1.6 * 1.6))
        img = (blobs[:, :, None] * col[:, :, :, None, None]).sum(1)
        frames.append(img.clamp(0, 1))
        positions.append(pos.clone())
        pos = pos + vel
        if kicks and t > 0 and t % 2 == 0:
            vel = vel + torch.randn(bs, nb, 2, generator=g) * 1.15
        if collisions:
            vel = vel.clone()
            for i in range(nb):
                for j in range(i + 1, nb):
                    rp = pos[:, j] - pos[:, i]
                    rv = vel[:, j] - vel[:, i]
                    hit = (rp.norm(dim=-1) < 3.2) & ((rv * rp).sum(-1) < 0)
                    if hit.any():
                        vi, vj = vel[:, i].clone(), vel[:, j].clone()
                        vel[:, i] = torch.where(hit[:, None], vj, vi)
                        vel[:, j] = torch.where(hit[:, None], vi, vj)
        low = pos < 0
        high = pos > maxs
        pos = torch.where(low, -pos, pos)
        pos = torch.where(high, 2 * maxs - pos, pos)
        vel = torch.where(low | high, -vel, vel)
    out = torch.stack(frames, 1)
    return out, {"pos": torch.stack(positions, 1), "col": col}, moving_mask(out)


def test_default_bytes():
    assert N_BALLS == 3 and RADIUS == 1.6 and SPEED == 1.15
    for seed in (0, 123, 170001):
        for kicks, collisions in ((False, False), (False, True), (True, True)):
            opts = dict(kicks=kicks, collisions=collisions)
            expected = _r11_default(2, 6, 16, 16, seed, **opts)
            actual = make_clip_batch(2, 6, 16, 16, seed=seed, kick_every=2,
                                     return_meta=True, return_moving=True, **opts)
            explicit = make_clip_batch(2, 6, 16, 16, seed=seed, nb=3, kick_every=2,
                                       return_meta=True, return_moving=True, **opts)
            for observed in (actual, explicit):
                assert torch.equal(observed[0], expected[0]), (seed, opts, "frames")
                assert torch.equal(observed[1]["pos"], expected[1]["pos"])
                assert torch.equal(observed[1]["col"], expected[1]["col"])
                assert torch.equal(observed[2], expected[2])
    print("OK default data: byte-identical to frozen R11 with collisions and kicks")


def test_larger_counts():
    global_rng = torch.random.get_rng_state().clone()
    for nb in (1, 3, 8, 12, 17):
        for grid, radius, speed in ((16, 1.6, 1.15), (32, 0.8, 2.30), (64, 1.6, 1.15)):
            opts = dict(nb=nb, collisions=True, seed=91, radius=radius, speed=speed,
                        return_meta=True, return_moving=True)
            frames, meta, moving = make_clip_batch(2, 6, grid, grid, **opts)
            again, again_meta, again_moving = make_clip_batch(2, 6, grid, grid, **opts)
            assert frames.shape == (2, 7, 3, grid, grid)
            assert meta["pos"].shape == (2, 7, nb, 2)
            assert meta["col"].shape == (2, nb, 3)
            assert moving.shape == (2, 6, grid, grid)
            assert torch.isfinite(frames).all() and 0 <= frames.min() <= frames.max() <= 1
            assert torch.all((meta["pos"] >= 0) & (meta["pos"] <= grid - 1))
            assert torch.equal(frames, again) and torch.equal(meta["pos"], again_meta["pos"])
            assert torch.equal(moving, again_moving)
            colors = meta["col"] / meta["col"].amax(-1, keepdim=True)
            nearest = (colors[:, :, None] - PALETTE_ANCHORS).abs().amax(-1)
            assert (nearest.min(-1).values < 1e-6).all(), "all colors use shared anchors"
            ids = nearest.argmin(-1)
            if nb > 3:
                for sample in ids:
                    counts = torch.bincount(sample, minlength=3)
                    assert counts.max() - counts.min() <= 1, "palette allocation is balanced"
    assert torch.equal(global_rng, torch.random.get_rng_state()), "local seed must not consume global RNG"
    for bad in (0, -2, 1.5, True, "8"):
        try:
            make_clip_batch(1, 1, 16, 16, nb=bad)
        except ValueError as exc:
            assert "positive integer" in str(exc)
        else:
            raise AssertionError(f"invalid nb accepted: {bad!r}")
    print("OK n-balls: K=1/3/8/12/17; grids 16/32/64; geometry; deterministic and bounded")


def test_large_index_collisions():
    # Isolate pair (10,11), checking both sample independence and the approach
    # gate. An implementation accidentally fixed at three balls would miss it.
    pos = torch.zeros(2, 12, 2)
    pos[:, :, 0] = torch.arange(12) * 20
    pos[:, 10, 0] = 300
    pos[:, 11, 0] = 301
    vel = torch.zeros_like(pos)
    vel[0, 10, 0], vel[0, 11, 0] = 1, -1
    vel[1, 10, 0], vel[1, 11, 0] = -1, 1
    before = vel.clone()
    resolved = _resolve_collisions(pos, vel)
    assert torch.equal(vel, before), "collision resolver must not mutate caller velocity"
    assert torch.equal(resolved[0, 10], before[0, 11])
    assert torch.equal(resolved[0, 11], before[0, 10])
    assert torch.equal(resolved[1], before[1]), "separating pair must not swap"
    assert torch.equal(resolved[:, :10], before[:, :10])
    assert torch.equal(resolved.sum(1), before.sum(1)), "momentum conserved"
    assert torch.equal(resolved.square().sum((1, 2)), before.square().sum((1, 2)))
    assert torch.equal(_resolve_collisions(pos, vel, contact=0.8), before), "radius scales contact"
    assert CONTACT == 2 * RADIUS
    coll = make_clip_batch(2, 16, 16, 16, nb=12, seed=7, collisions=True, return_meta=True)[1]
    free = make_clip_batch(2, 16, 16, 16, nb=12, seed=7, collisions=False, return_meta=True)[1]
    assert not torch.equal(coll["pos"], free["pos"]), "generator must engage collisions at K=12"
    print("OK collisions: high-index pair exchanges velocity; energy/momentum and radius contact preserved")


if __name__ == "__main__":
    torch.set_num_threads(1)
    test_default_bytes()
    test_larger_counts()
    test_large_index_collisions()
