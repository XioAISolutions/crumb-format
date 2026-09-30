#!/usr/bin/env python3
"""BrainsNN Studio v0 — local prompt -> long video generator.

Anyone on this Mac: open http://127.0.0.1:9321 , type a scene, pick a length,
press Generate. The 4090 on the box renders it; the video appears here.
Stdlib only. Orchestrates via ssh/scp to the GPU box.
"""
import json
import re
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "config.json").read_text())
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
PORT = 9321

SSH = ["ssh", "-p", str(CFG["port"]), f'{CFG["user"]}@{CFG["host"]}']
SCP = ["scp", "-q", "-P", str(CFG["port"])]
REMOTE = f'{CFG["user"]}@{CFG["host"]}'

_jobs = {}
_lock = threading.Lock()

_pc = ROOT / "passcode.txt"
CODE = _pc.read_text().strip() if _pc.exists() else ""

_STATE = ROOT / "jobs.json"


def _save():
    try:
        _STATE.write_text(json.dumps(list(_jobs.values())))
    except Exception:
        pass


def _load():
    if _STATE.exists():
        try:
            for j in json.loads(_STATE.read_text()):
                _jobs[j["id"]] = j
        except Exception:
            pass


_load()


def sh(args, timeout=60):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except Exception as e:  # noqa: BLE001
        return 1, "", str(e)


def ssh(cmd, timeout=60):
    return sh(SSH + [cmd], timeout=timeout)


JOB_TMPL = """#!/bin/bash
set -e
cd {exp}
exec > {box}/logs/gpuq_{jid}.log 2>&1
export PY={box}/venvs/longlive/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LL_PROMPT_CHUNK=16 LL_OFFLOAD_IDLE=1 LL=/root/LongLive
export JOB_ID={jid} OUT=runs_{jid} LENGTHS={mins} PRECISION=fp8 WINDOW=12 SINK=8 SEED={seed} RELATIVE_ROPE=on
export PROMPTS={box}/prompts_{jid}.txt
exec /usr/bin/timeout {tmo} bash run_longlive.sh
"""


def submit(prompt, minutes, seed):
    jid = "studio_" + time.strftime("%y%m%d%H%M%S") + "_" + uuid.uuid4().hex[:3]
    tmo = max(2400, int(minutes) * 25)
    script = JOB_TMPL.format(jid=jid, mins=minutes, seed=seed, tmo=tmo,
                             exp=CFG["box_exp"], box=CFG["box_dir"])
    stage = ROOT / "stage"
    stage.mkdir(exist_ok=True)
    (stage / f"prompts_{jid}.txt").write_text(prompt.strip() + "\n")
    (stage / f"{jid}.sh").write_text(script)
    rc1, _, e1 = sh(SCP + [str(stage / f"prompts_{jid}.txt"),
                           str(stage / f"{jid}.sh"), f"{REMOTE}:{CFG['box_dir']}/"])
    rc2, o2, e2 = ssh("cd {0} && cp {1}.sh gpu_queue/pending/gpuq_job_974zzzzzz_{1}.sh && echo OK".format(CFG["box_dir"], jid), 60)
    ok = (rc1 == 0 and "OK" in o2)
    with _lock:
        _jobs[jid] = {"id": jid, "prompt": prompt.strip(), "minutes": minutes,
                      "seed": seed, "status": "queued" if ok else "submit_failed",
                      "error": "" if ok else (e1 + " " + e2).strip()[:300],
                      "created": time.time(), "video": None}
        _save()
    return jid


def _finished(jid):
    _, o, _ = ssh('grep "FINISHED gpuq_job_974zzzzzz_%s" %s/logs/gpuq.log | tail -1'
                  % (jid, CFG["box_dir"]), 30)
    if not o.strip():
        return None
    m = re.search(r"rc=(-?\d+) dur=(\d+)s", o)
    return (int(m.group(1)), int(m.group(2))) if m else (1, 0)


def step(j):
    jid = j["id"]
    if j["status"] == "queued":
        _, o, _ = ssh('grep -c "RUNNING gpuq_job_974zzzzzz_%s" %s/logs/gpuq.log'
                      % (jid, CFG["box_dir"]), 30)
        if o.strip().isdigit() and int(o.strip()) > 0:
            j["status"] = "running"
            j["started"] = time.time()
        elif _finished(jid) is not None:
            j["status"] = "running"
    if j["status"] == "running":
        fin = _finished(jid)
        if fin is not None:
            rc, dur = fin
            if rc == 0:
                j["status"] = "fetching"
                j["duration_s"] = dur
            else:
                j["status"] = "failed"
                j["error"] = "generation rc=%d" % rc
    if j["status"] == "fetching":
        d = "%s/exp/pr63/runs_%s/len_%ss" % (CFG["box_dir"], jid, j["minutes"])
        dst = OUT / (jid + ".mp4")
        rc, _, e = sh(SCP + ["%s:%s/rank0-0-0_regular.mp4" % (REMOTE, d), str(dst)], 900)
        if rc == 0 and dst.exists():
            prev = OUT / (jid + ".preview.mp4")
            sh(["ffmpeg", "-y", "-v", "error", "-i", str(dst), "-vf", "scale=960:-2",
                "-c:v", "libx264", "-crf", "30", "-preset", "veryfast", "-an", str(prev)], 1200)
            j["status"] = "done"
            j["video"] = "/video/" + jid
        else:
            j["status"] = "fetch_failed"
            j["error"] = e[:300]


def worker():
    while True:
        with _lock:
            active = [x for x in _jobs.values()
                      if x["status"] in ("queued", "running", "fetching")]
        for x in active:
            try:
                step(x)
            except Exception as e:  # noqa: BLE001
                x["error"] = str(e)[:300]
        if active:
            _save()
        time.sleep(15)


PASS_PAGE = (b"<!doctype html><html><head><meta charset=utf-8>"
             b"<title>BrainsNN Studio</title></head><body style='font-family:system-ui;"
             b"background:#0b0d10;color:#e8eaed;display:grid;place-items:center;height:100vh'>"
             b"<form method=get><h3>BrainsNN Studio</h3>"
             b"<input name=k autofocus placeholder='passcode' style='padding:10px;font-size:16px;border-radius:8px;border:1px solid #333;background:#111;color:#eee'> "
             b"<button style='padding:10px 16px;font-size:16px;border-radius:8px;border:0;background:#2f6fed;color:#fff'>Open</button>"
             b"</form></body></html>")


def _authcheck(handler):
    handler._set_cookie = None
    if not CODE:
        return True
    from http.cookies import SimpleCookie
    c = SimpleCookie(handler.headers.get("Cookie", ""))
    if c.get("sbk") and c["sbk"].value == CODE:
        return True
    if ("k=" + CODE) in handler.path:
        handler._set_cookie = "sbk=%s; Path=/; Max-Age=43200" % CODE
        return True
    return False


class H(BaseHTTPRequestHandler):
    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if getattr(self, "_set_cookie", None):
            self.send_header("Set-Cookie", self._set_cookie)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass

    def do_GET(self):
        if not _authcheck(self):
            return self._send(401, "text/html; charset=utf-8", PASS_PAGE)
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"):
            return self._send(200, "text/html; charset=utf-8", (ROOT / "index.html").read_bytes())
        if p == "/api/jobs":
            with _lock:
                jobs = sorted(_jobs.values(), key=lambda x: -x["created"])
            return self._send(200, "application/json", json.dumps(jobs).encode())
        if p.startswith("/video/"):
            jid = p.split("/")[-1]
            f = OUT / (jid + ".preview.mp4")
            if not f.exists():
                f = OUT / (jid + ".mp4")
            if f.exists():
                return self._send(200, "video/mp4", f.read_bytes())
            return self._send(404, "text/plain", b"not found")
        if p.startswith("/raw/"):
            f = OUT / (p.split("/")[-1] + ".mp4")
            if f.exists():
                return self._send(200, "video/mp4", f.read_bytes())
            return self._send(404, "text/plain", b"not found")
        return self._send(404, "text/plain", b"no")

    def do_POST(self):
        if not _authcheck(self):
            return self._send(401, "text/plain", b"auth")
        if self.path != "/api/generate":
            return self._send(404, "text/plain", b"no")
        try:
            n = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(n) or b"{}")
            prompt = (data.get("prompt") or CFG["default_prompt"]).strip()
            minutes = int(data.get("minutes") or 10)
            seed = int(data.get("seed") or 0)
            if not (4 <= minutes <= 360):
                raise ValueError("length out of range")
            jid = submit(prompt, minutes, seed)
            return self._send(200, "application/json", json.dumps({"id": jid}).encode())
        except Exception as e:  # noqa: BLE001
            return self._send(400, "text/plain", str(e).encode())


def main():
    threading.Thread(target=worker, daemon=True).start()
    print("BrainsNN Studio on http://127.0.0.1:%d" % PORT)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()


if __name__ == "__main__":
    main()
