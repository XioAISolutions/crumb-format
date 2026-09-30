import cv2, numpy as np, sys, json, statistics as st
p = sys.argv[1] if len(sys.argv) > 1 else 'preview_30s.mp4'
cap = cv2.VideoCapture(p)
prev = None; idx = 0; mags = []; dxs = []; rmses = []
while True:
    ok, fr = cap.read()
    if not ok:
        break
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (640, 352))
    if prev is not None:
        flow = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        mags.append(float(np.median(mag)))
        dxs.append(float(np.median(flow[..., 0])))
        rmses.append(float(np.sqrt(np.mean((g.astype(np.float32) - prev.astype(np.float32))**2))))
    prev = g; idx += 1
cap.release()
m = np.array(mags); dx = np.array(dxs); rm = np.array(rmses)
out = {
    'frames': idx,
    'flow_med_median_px': round(st.median(mags), 3),
    'flow_med_p10': round(float(np.percentile(m, 10)), 3),
    'flow_med_p90': round(float(np.percentile(m, 90)), 3),
    'dx_median_px': round(st.median(dxs), 3),
    'dx_sign_frac_neg': round(float((dx < 0).mean()), 3),
    'net_dx_sum_px': round(float(dx.sum()), 1),
    'rmse_median': round(st.median(rmses), 2),
    'rmse_p90_over_median': round(float(np.percentile(rm, 90) / max(st.median(rmses), 1e-6)), 2),
}
n = len(m); b = 8
if n >= 32:
    lim = min(n, 300)
    intra = [m[i] for i in range(1, lim) if i % b != 0]
    bound = [m[i] for i in range(1, lim) if i % b == 0]
    out['intra8_med'] = round(st.median(intra), 3)
    out['bound8_med'] = round(st.median(bound), 3)
    out['bound_over_intra'] = round(st.median(bound) / max(st.median(intra), 1e-9), 2)
half = n // 2
out['flow_first_half'] = round(st.median(m[:half]), 3)
out['flow_second_half'] = round(st.median(m[half:]), 3)
print(json.dumps(out, indent=1))
