#!/usr/bin/env python3
"""Wave-Field 2D (spacetime) mixing — smoke test.

1) Separable 2D damped-cosine kernel + FFT circular conv
2) Numerical equivalence: FFT conv == brute-force circular conv
3) Causal time masking still equivalent
4) Scaling: wave-field conv cost vs full attention cost at equal N
"""
import time, json, pathlib
import torch

torch.manual_seed(7)

def wave_kernel_2d(at, asc, wt, ws, pt, psc, T, S):
    dt = (torch.arange(T) - T // 2).float()
    ds = (torch.arange(S) - S // 2).float()
    kt = torch.exp(-at * dt.abs()) * torch.cos(wt * dt + pt)
    ks = torch.exp(-asc * ds.abs()) * torch.cos(ws * ds + psc)
    return torch.outer(kt, ks)

def fft_conv2d(x, k):
    K = torch.fft.rfft2(torch.fft.ifftshift(k))
    X = torch.fft.rfft2(x)
    return torch.fft.irfft2(X * K, s=x.shape[-2:])

def direct_circular_conv2d(x, k):
    kc = torch.fft.ifftshift(k)
    out = torch.zeros_like(x)
    T, S = x.shape[-2:]
    for u in range(T):
        for v in range(S):
            out = out + kc[u, v] * torch.roll(x, shifts=(u, v), dims=(-2, -1))
    return out

def equivalence_check():
    T = S = 16
    x = torch.randn(1, 1, T, S)
    k = wave_kernel_2d(0.08, 0.08, 0.9, 0.6, 0.2, 0.1, T, S)
    a = fft_conv2d(x, k)
    b = direct_circular_conv2d(x, k)
    err = (a - b).abs().max().item()
    kt_causal = k.clone()
    dt = (torch.arange(T) - T // 2)
    kt_causal[dt > 0, :] = 0.0          # zero the future half along time
    c = fft_conv2d(x, kt_causal)
    d = direct_circular_conv2d(x, kt_causal)
    err_c = (c - d).abs().max().item()
    return err, err_c

def attention_time(x):
    B, C, T, S = x.shape
    N = T * S
    q = x.reshape(B, C, N).transpose(1, 2)
    start = time.perf_counter()
    att = torch.matmul(q, q.transpose(-1, -2)) / (C ** 0.5)
    out = torch.matmul(att, q)
    return time.perf_counter() - start

def wave_time(x):
    T, S = x.shape[-2:]
    k = wave_kernel_2d(0.02, 0.02, 0.5, 0.5, 0.0, 0.0, T, S)
    start = time.perf_counter()
    y = fft_conv2d(x, k)
    return time.perf_counter() - start

def main():
    err, err_c = equivalence_check()
    print("EQ max err (circular):", err)
    print("EQ max err (causal)  :", err_c)
    assert err < 1e-4 and err_c < 1e-4, "equivalence failed"
    rows = []
    for grid in [32, 64, 128, 256]:
        T = S = grid
        x = torch.randn(1, 64, T, S)
        wt = wave_time(x)
        at = attention_time(x) if grid <= 128 else float("nan")
        rows.append({"T": T, "S": S, "N": T * S, "wave_ms": round(wt * 1000, 2), "attn_ms": (round(at * 1000, 2) if at == at else "skipped(OOM-risk)")})
        print(json.dumps(rows[-1]))
    out = pathlib.Path(__file__).parent / "RESULTS.md"
    lines = ["# Wave-Field 2D smoke test", "",
             f"- equivalence max err: {err:.2e} (plain), {err_c:.2e} (causal)",
             "- scaling (single 64-channel 2D field, CPU):", ""]
    for r in rows:
        lines.append(f"- {r['T']}x{r['S']} (N={r['N']}): wave={r['wave_ms']} ms, attention={r['attn_ms']}")
    lines += ["", "Next: bilinear 2D scatter/gather + a tiny video-prediction", "training run (wave-field vs transformer baseline)."]
    out.write_text("\n".join(lines) + "\n")
    print("WROTE", out)

if __name__ == "__main__":
    main()
