"""Bounded offline CPU-only LTX encoder probe of an existing publisher-pinned cache.

This is a partial pretrained encoder contract probe, not the production DINO/
decoder/quality preflight, learned training, or a GPU result. No downloads/copies.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import struct
import subprocess
import sys
import time


def write(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def worker(arguments):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    from validation import (GateError, load_pretrained_vae, probe_raw_future_causality,
                            sha256_file)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = Path(arguments.out)
    started = time.monotonic()
    state = {"status": "running", "stage": "verify_publisher_pin", "pretrained_evidence": False,
             "GPU_evidence": False, "trained_model_evidence": False, "quality_claim_allowed": False,
             "native_resolution_quality_evidence": False, "device": "cpu", "threads": 1,
             "raw_probe_shape": [1, 297, 3, 32, 32], "raw_prefix_cuts": [33, 129, 289],
             "scope": "synthetic low-resolution raw encoder causality at full learned temporal horizon only",
             "remaining_stages": {"pretrained_decoder_rf_and_chunking": "unrun",
                                  "frozen_DINO_features": "blocked_transformers_missing",
                                  "native_footage_quality": "unverified",
                                  "learned_three_seed_campaign": "unrun"}}
    progress_path = output / "progress.json"
    try:
        publisher_path = Path(arguments.publisher_metadata)
        publisher = json.loads(publisher_path.read_text())
        revision = publisher["revision"]
        expected_url = f"https://huggingface.co/api/models/Lightricks/LTX-Video/revision/{revision}?blobs=true"
        if publisher.get("repository") != "Lightricks/LTX-Video" or publisher.get("publisher_url") != expected_url:
            raise GateError("publisher metadata repository/revision URL does not match the pinned LTX snapshot")
        root = Path(arguments.vae_dir).resolve()
        if root.name != "vae" or root.parent.name != revision:
            raise GateError("cached component path differs from publisher-pinned snapshot revision")
        config_path, weight_path = root / "config.json", root / "diffusion_pytorch_model.safetensors"
        file_metadata = publisher["vae_files"]
        config_meta = file_metadata["vae/config.json"]
        weight_meta = file_metadata["vae/diffusion_pytorch_model.safetensors"]
        config_bytes = config_path.read_bytes()
        blob_id = hashlib.sha1(f"blob {len(config_bytes)}\0".encode()+config_bytes).hexdigest()
        if len(config_bytes) != config_meta["size"] or blob_id != config_meta["blobId"]:
            raise GateError("local config differs from publisher git-blob identity")
        if weight_path.stat().st_size != weight_meta["size"]:
            raise GateError("cached pretrained weight size differs from publisher")
        state.update({"repository": publisher["repository"], "revision": revision,
                      "publisher_metadata_sha256": sha256_file(publisher_path), "stage": "stream_hash_existing_weights"})
        write(progress_path, state)
        weight_hash = sha256_file(weight_path)
        if weight_hash != weight_meta["lfs"]["sha256"]:
            raise GateError("cached weight bytes differ from official publisher LFS SHA256")
        manifest = {"backend": "ltx", "repository": publisher["repository"], "revision": revision,
                    "source_kind": "official_huggingface_snapshot",
                    "files": {"config.json": hashlib.sha256(config_bytes).hexdigest(),
                              "diffusion_pytorch_model.safetensors": weight_hash}}
        manifest_path = output / "publisher_snapshot_manifest.json"
        write(manifest_path, manifest)
        state.update({"publisher_bytes_verified": True, "weight_sha256": weight_hash,
                      "weight_bytes": weight_path.stat().st_size, "stage": "load_frozen_pretrained_cpu_VAE"})
        write(progress_path, state)
        vae, proof = load_pretrained_vae("ltx", root, manifest_path, sha256_file(manifest_path), device="cpu")
        # Refuse version-compatibility loading that would initialize extra random
        # parameters rather than loading this complete pretrained checkpoint.
        with weight_path.open("rb") as handle:
            length = struct.unpack("<Q", handle.read(8))[0]
            if length > 1 << 20:
                raise GateError("safetensors header exceeds bounded inspection")
            header = json.loads(handle.read(length))
        header.pop("__metadata__", None)
        loaded = vae.model.state_dict()
        if set(loaded) != set(header) or any(list(loaded[key].shape) != header[key]["shape"] for key in loaded):
            raise GateError("loaded native model state differs from publisher checkpoint; random/new parameters refused")
        state.update({"pretrained_evidence": True, "vae": proof, "loaded_state_keys_exact": True,
                      "stage": "raw_future_poison_full297_cut33_129_289"})
        write(progress_path, state)
        frames = .2+.6*torch.rand(1, 297, 3, 32, 32, generator=torch.Generator().manual_seed(19013))
        result = probe_raw_future_causality(vae, frames, (33, 129, 289), required_context_frames=289)
        state.update({"status": "pretrained_cpu_encoder_contract_pass" if result["pass"] else "pretrained_cpu_encoder_contract_reject",
                      "stage": "complete_partial_encoder_probe", "encoder_causality": result,
                      "production_preflight_complete": False, "torch": torch.__version__,
                      "diffusers": importlib.metadata.version("diffusers")})
    except Exception as error:
        state.update({"status": "pretrained_cpu_encoder_probe_failed", "error_type": type(error).__name__,
                      "error": str(error), "production_preflight_complete": False})
    state["worker_wall_seconds"] = time.monotonic()-started
    write(output / "worker_result.json", state)
    write(progress_path, state)
    return 0 if state["status"] == "pretrained_cpu_encoder_contract_pass" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--publisher-metadata", required=True)
    parser.add_argument("--vae-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", "--publisher-metadata",
               args.publisher_metadata, "--vae-dir", args.vae_dir, "--out", str(output)]
    started = time.monotonic()
    cap_reason, peak_rss_kib = None, 0
    with (output / "worker_stdout.log").open("w") as stdout, (output / "worker_stderr.log").open("w") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
        while process.poll() is None:
            rss = subprocess.run(["ps", "-o", "rss=", "-p", str(process.pid)], capture_output=True, text=True)
            try:
                peak_rss_kib = max(peak_rss_kib, int(rss.stdout.strip()))
            except ValueError:
                pass
            if time.monotonic()-started > 180:
                cap_reason = "hard_180_second_process_group_timeout"
            elif peak_rss_kib > 8*1024*1024:
                cap_reason = "hard_8GiB_RSS_process_group_cap"
            if cap_reason:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                break
            time.sleep(.25)
    result_path = output / "worker_result.json"
    if result_path.is_file():
        report = json.loads(result_path.read_text())
    elif (output / "progress.json").is_file():
        report = json.loads((output / "progress.json").read_text())
    else:
        report = {"pretrained_evidence": False, "GPU_evidence": False, "quality_claim_allowed": False}
    if cap_reason:
        report.update({"status": "pretrained_cpu_encoder_probe_capped", "cap_reason": cap_reason})
    elif "status" not in report or report["status"] == "running":
        report.update({"status": "worker_failed_before_result", "exit_code": process.returncode})
    from validation import NATIVE_ROOT, sha256_file
    report.update({"outer_wall_seconds": time.monotonic()-started, "peak_worker_rss_kib": peak_rss_kib,
                   "outer_timeout_seconds": 180, "outer_rss_cap_GiB": 8,
                   "production_preflight_complete": False,
                   "source_sha256": {name: sha256_file(Path(__file__).parent/name) for name in
                                     ("run_cached_ltx_cpu.py", "validation.py")},
                   "native_source_sha256": {"video_vae.py": sha256_file(NATIVE_ROOT / "video_vae.py")},
                   "platform": platform.platform()})
    write(output / "receipt.json", report)
    print(json.dumps({"status": report["status"], "receipt": str(output / "receipt.json"),
                      "pretrained_evidence": report["pretrained_evidence"], "GPU_evidence": report["GPU_evidence"]}))
    return 0 if report["status"] == "pretrained_cpu_encoder_contract_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
