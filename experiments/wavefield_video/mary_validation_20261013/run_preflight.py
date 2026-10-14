"""Offline pretrained-VAE validation. Default CUDA is required, never silently replaced.

--smoke runs labelled arithmetic CPU fixtures only. It cannot produce pretrained,
GPU, learned-recall, video-quality, or architectural-superiority evidence.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch

from validation import (FairProtocol, FrozenImageFeatures, GateError, NATIVE_ROOT,
                        decoded_health_report, load_pretrained_vae, pixel_metrics,
                        probe_decoder_chunking, probe_raw_future_causality, sha256_file)


class _SmokeCodec(torch.nn.Module):
    t_stride = 4

    def __init__(self, future_pool=False):
        super().__init__()
        self.future_pool = future_pool
        self.eval()

    def encode(self, frames):
        sampled = frames[:, ::4].clone()
        return sampled+frames.mean(1, keepdim=True) if self.future_pool else sampled

    def decode(self, latents):
        return torch.cat((latents[:, :1], latents[:, 1:].repeat_interleave(4, dim=1)), 1)


def _smoke():
    frames = .2+.6*torch.rand(1, 33, 3, 8, 8, generator=torch.Generator().manual_seed(19013))
    causal = probe_raw_future_causality(_SmokeCodec(), frames, (1, 9, 17))
    negative = probe_raw_future_causality(_SmokeCodec(True), frames, (1, 9, 17))
    decoder = probe_decoder_chunking(_SmokeCodec(), 3, 4, 4, max_rf_latents=40)
    generated = frames[:, :4].clone()
    generated[:, 2:] = 0
    health = decoded_health_report(frames[:, :9], generated, stride=4, patience=2)
    return {"status": "cpu_contract_smoke_pass" if causal["pass"] and not negative["pass"] and decoder["pass"] else "cpu_contract_smoke_fail",
            "synthetic_arithmetic_fixture": True, "pretrained_evidence": False, "GPU_evidence": False,
            "causal_fixture": causal, "acausal_negative_fixture": negative, "decoder_fixture": decoder,
            "native_pixel_fixture_health": health,
            "frozen_feature_evidence": False, "learned_recall_evidence": False, "quality_claim_allowed": False}


def _raw_probe(arguments):
    if arguments.probe_tensor:
        raw = torch.load(arguments.probe_tensor, map_location="cpu", weights_only=True)
        if isinstance(raw, dict):
            if set(raw) != {"frames"}:
                raise GateError("probe tensor packs contain frames only; targets/labels are not encoder inputs")
            raw = raw["frames"]
        if not isinstance(raw, torch.Tensor):
            raise GateError("probe tensor must be a raw RGB frame tensor")
        return raw, {"file_sha256": sha256_file(arguments.probe_tensor), "source_kind": "owned_raw_tensor"}
    import imageio.v2 as imageio
    reader = imageio.get_reader(arguments.probe_video, "ffmpeg")
    native_fps = reader.get_meta_data().get("fps")
    frames = []
    try:
        for frame in reader:
            frames.append(torch.from_numpy(frame.copy()).permute(2, 0, 1).float()/255)
            if len(frames) == arguments.probe_raw_frames:
                break
    finally:
        reader.close()
    if len(frames) != arguments.probe_raw_frames:
        raise GateError("owned raw video is shorter than the registered causal probe")
    return torch.stack(frames)[None], {"file_sha256": sha256_file(arguments.probe_video),
                                      "source_kind": "owned_native_video", "spatial_resize": False,
                                      "native_fps": native_fps}


def _production(arguments, progress):
    progress["stage"] = "local_artifact_prerequisites"
    for value in (arguments.vae_dir, arguments.vae_manifest, arguments.vae_manifest_sha256,
                  arguments.feature_dir, arguments.feature_manifest, arguments.feature_manifest_sha256):
        if not value:
            raise GateError("production preflight requires existing pinned pretrained VAE AND frozen feature artifacts")
    if bool(arguments.probe_tensor) == bool(arguments.probe_video):
        raise GateError("supply exactly one owned raw probe video or tensor")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    progress["stage"] = "load_frozen_pretrained_vae"
    vae, proof = load_pretrained_vae(arguments.backend, arguments.vae_dir, arguments.vae_manifest,
                                   arguments.vae_manifest_sha256, device=arguments.device)
    progress.update({"pretrained_evidence": True, "GPU_evidence": arguments.device.startswith("cuda"),
                     "evidence_scope": "codec preflight only; no learned GPU campaign", "vae": proof,
                     "stage": "owned_raw_probe"})
    required_context = FairProtocol().mapping(vae.t_stride)["observed_raw_frames_per_arm"]
    arguments.probe_raw_frames = arguments.probe_raw_frames or required_context+vae.t_stride
    arguments.cuts = arguments.cuts or (1, 33, 129, required_context)
    if arguments.probe_raw_frames < required_context+vae.t_stride or not {33, 129, required_context} <= set(arguments.cuts):
        raise GateError("production raw probe must cover 289 context frames plus an unseen block, cuts 33/129/289")
    progress["registered_probe_raw_frames"] = arguments.probe_raw_frames
    progress["registered_cuts"] = list(arguments.cuts)
    frames, source = _raw_probe(arguments)
    progress["raw_probe_source"] = source
    if frames.shape[1] != arguments.probe_raw_frames:
        raise GateError("probe frame count differs from the registered value; no implicit truncation")
    progress["stage"] = "raw_encoder_future_causality"
    causality = probe_raw_future_causality(vae, frames, arguments.cuts,
                                         required_context_frames=required_context)
    progress["causality"] = causality
    base = {"pretrained_evidence": True, "GPU_evidence": arguments.device.startswith("cuda"),
            "vae": proof, "raw_probe_source": source, "causality": causality,
            "fair_horizon_mapping": FairProtocol().mapping(vae.t_stride),
            "learned_recall_evidence": False, "quality_claim_allowed": False,
            "architecture_superiority_claim_allowed": False}
    if not causality["pass"]:
        return {**base, "status": "rejected_raw_encoder_future_dependence", "frozen_feature_evidence": False,
                "training_allowed": False}
    progress["stage"] = "actual_weight_decoder_rf_chunk_and_resume"
    latent = vae.encode(frames)
    decoder = probe_decoder_chunking(vae, vae.channels, latent.shape[-2], latent.shape[-1])
    progress["decoder"] = decoder
    if not decoder["pass"]:
        return {**base, "status": "rejected_decoder_chunk_or_seam_mismatch", "decoder": decoder,
                "frozen_feature_evidence": False, "training_allowed": False}
    progress["stage"] = "load_pinned_frozen_image_features"
    feature = FrozenImageFeatures(arguments.feature_dir, arguments.feature_manifest,
                                  arguments.feature_manifest_sha256, device=arguments.device)
    progress["frozen_feature_identity"] = feature.proof
    with torch.no_grad():
        decoded = vae.decode(latent).cpu()
    if decoded.shape != frames.shape:
        raise GateError("VAE reconstruction changed the native raw grid; no upsample/crop quality shortcut")
    progress["stage"] = "native_decoded_pixel_and_fixed_context_health"
    health = decoded_health_report(frames[:, :9], decoded[:, 9:], fps=source.get("native_fps") or arguments.fps,
                                   stride=vae.t_stride, features=feature)
    health["clean_context_trust"] = "unreviewed raw probe prefix; independent native-pixel validation required"
    return {**base, "status": "pretrained_vae_preflight_complete_not_quality_certification",
            "decoded_reconstruction_pixels": pixel_metrics(decoded, frames), "health": health, "decoder": decoder,
            "frozen_feature_evidence": True, "training_allowed": False,
            "next_gate": "native blind challenge, all required stronger baselines, matched write/bias/resources before learned campaign"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", required=True, help="fresh output directory; never overwrite prior evidence")
    parser.add_argument("--backend", choices=("wan", "ltx"), default="wan")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vae-dir")
    parser.add_argument("--vae-manifest")
    parser.add_argument("--vae-manifest-sha256")
    parser.add_argument("--feature-dir")
    parser.add_argument("--feature-manifest")
    parser.add_argument("--feature-manifest-sha256")
    parser.add_argument("--probe-video")
    parser.add_argument("--probe-tensor")
    parser.add_argument("--probe-raw-frames", type=int, default=None,
                        help="production minimum: 289 observed frames + one unseen VAE stride block")
    parser.add_argument("--cuts", type=int, nargs="+", default=None)
    parser.add_argument("--fps", type=float, default=24)
    arguments = parser.parse_args(argv)
    output = Path(arguments.out)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(1)
    progress = {"pretrained_evidence": False, "GPU_evidence": False}
    try:
        report = _smoke() if arguments.smoke else _production(arguments, progress)
    except Exception as error:
        report = {**progress, "status": "preflight_failed", "error_type": type(error).__name__, "error": str(error),
                  "quality_claim_allowed": False,
                  "learned_recall_evidence": False, "training_allowed": False}
    report.update({"wall_seconds": time.monotonic()-started, "torch": torch.__version__,
                   "command_argv": list(argv) if argv is not None else sys.argv[1:],
                   "source_sha256": {name: sha256_file(Path(__file__).parent/name) for name in
                                     ("validation.py", "run_preflight.py", "launch_protocol.json")},
                   "native_dependency_sha256": {name: sha256_file(NATIVE_ROOT/name) for name in
                                                ("video_vae.py", "long_horizon.py")}})
    (output / "receipt.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": report["status"], "receipt": str(output / "receipt.json"),
                      "pretrained_evidence": report["pretrained_evidence"], "GPU_evidence": report["GPU_evidence"]}))
    return 0 if report["status"] in ("cpu_contract_smoke_pass", "pretrained_vae_preflight_complete_not_quality_certification") else 1


if __name__ == "__main__":
    raise SystemExit(main())
