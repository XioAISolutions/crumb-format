"""Preflight gates, not a model/quality promotion or a download mechanism."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import torch
import torch.nn.functional as F

NATIVE_ROOT = Path(__file__).resolve().parent.parent
if str(NATIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(NATIVE_ROOT))
from long_horizon import HealthMonitor
from video_vae import StreamDecoder, history_with_margin, measure_temporal_rf

REPOSITORIES = {
    "wan": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "ltx": "Lightricks/LTX-Video",
    "dinov2": "facebook/dinov2-small",
}


class GateError(ValueError):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(str((tuple(value.shape), value.dtype)).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verify_local_manifest(directory, manifest_path, approved_sha256, backend):
    """Require independently pinned provenance and every local loading artifact.

    Hashes prove identity/integrity, not that training happened. The approved
    manifest is the operator's provenance trust anchor; no self-reported backend,
    config name, or arbitrary renamed tiny checkpoint suffices without that pin.
    """
    if backend not in REPOSITORIES:
        raise GateError("only production wan/ltx and pinned dinov2 artifacts are allowed; no conv/tiny")
    if not re.fullmatch(r"[0-9a-f]{64}", approved_sha256 or ""):
        raise GateError("an independently approved manifest SHA256 is required")
    root, manifest_path = Path(directory).expanduser().resolve(), Path(manifest_path)
    if not root.is_dir() or not manifest_path.is_file():
        raise GateError("local snapshot and provenance manifest must already exist; downloads are disabled")
    if sha256_file(manifest_path) != approved_sha256:
        raise GateError("provenance manifest differs from the approved pin")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("backend") != backend or manifest.get("repository") != REPOSITORIES[backend]
            or manifest.get("source_kind") != "official_huggingface_snapshot"
            or not re.fullmatch(r"[0-9a-f]{40}", manifest.get("revision", ""))):
        raise GateError("production backend/repository/pinned revision provenance is missing or mismatched")
    files = manifest.get("files", {})
    if not isinstance(files, dict) or "config.json" not in files:
        raise GateError("manifest must pin config.json and all weight/loading files")
    loading = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
               and (path.name == "config.json" or path.suffix in (".safetensors", ".bin")
                    or path.name.endswith(".index.json"))}
    if not any(name.endswith((".safetensors", ".bin")) for name in loading) or loading != set(files):
        raise GateError("manifest must cover exactly all config, weight, and shard-index files")
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise GateError("invalid local artifact manifest entry")
        if sha256_file(root / relative) != expected:
            raise GateError(f"local pretrained artifact changed: {name}")
        if name.endswith(".index.json"):
            index = json.loads((root / relative).read_text())
            mapping = index.get("weight_map")
            if not isinstance(mapping, dict) or not mapping:
                raise GateError("shard index must declare its hashed weight files")
            for shard in mapping.values():
                if not isinstance(shard, str):
                    raise GateError("invalid shard index filename")
                target = Path(shard)
                if (target.is_absolute() or ".." in target.parts or target.as_posix() not in files
                        or target.suffix not in (".safetensors", ".bin")):
                    raise GateError("shard index references an unhashed or escaping weight file")
    config = json.loads((root / "config.json").read_text())
    if backend in ("wan", "ltx"):
        channel_key, channels = ("z_dim", 16) if backend == "wan" else ("latent_channels", 128)
        class_name = "AutoencoderKLWan" if backend == "wan" else "AutoencoderKLLTXVideo"
        if config.get(channel_key) != channels or config.get("_class_name") != class_name:
            raise GateError("snapshot architecture does not match the production preset; tiny/random configs rejected")
    elif config.get("model_type") != "dinov2":
        raise GateError("frozen feature snapshot must be DINOv2")
    return {"backend": backend, "repository": manifest["repository"], "revision": manifest["revision"],
            "approved_manifest_sha256": approved_sha256, "artifacts": files,
            "provenance_verified": True,
            "provenance_scope": "approved publisher snapshot identity; hashes alone do not prove training"}


def assert_frozen(module):
    if not isinstance(module, torch.nn.Module):
        raise GateError("a real frozen torch module is required")
    if any(part.training for part in module.modules()) or any(p.requires_grad for p in module.parameters()):
        raise GateError("encoder must remain in eval mode with every parameter frozen")


def load_pretrained_vae(backend, directory, manifest, approved_sha256, *, device="cuda"):
    proof = verify_local_manifest(directory, manifest, approved_sha256, backend)
    if backend not in ("wan", "ltx"):
        raise GateError("VAE backend must be production wan or ltx")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise GateError("CUDA is unavailable; production preflight refuses automatic CPU fallback")
    from video_vae import VideoVAE
    vae = VideoVAE(backend, path=str(Path(directory).resolve()), device=device).eval().requires_grad_(False)
    assert_frozen(vae)
    proof.update(vae.describe())
    proof["weight_parameters"] = sum(p.numel() for p in vae.parameters())
    return vae, proof


class FrozenImageFeatures(torch.nn.Module):
    def __init__(self, directory, manifest, approved_sha256, *, device="cuda"):
        super().__init__()
        self.proof = verify_local_manifest(directory, manifest, approved_sha256, "dinov2")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise GateError("CUDA is unavailable; no automatic feature-model fallback")
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(str(Path(directory).resolve()), local_files_only=True,
                                               trust_remote_code=False).to(device).eval().requires_grad_(False)
        self.eval().requires_grad_(False)
        self.device = torch.device(device)
        self.register_buffer("mean", torch.tensor([.485, .456, .406], device=device)[None, :, None, None])
        self.register_buffer("std", torch.tensor([.229, .224, .225], device=device)[None, :, None, None])
        assert_frozen(self)

    @torch.no_grad()
    def forward(self, frames):
        x = F.interpolate(frames.to(self.device), size=(224, 224), mode="bilinear",
                          antialias=True, align_corners=False)
        result = self.model(pixel_values=(x-self.mean)/self.std)
        value = result.pooler_output if getattr(result, "pooler_output", None) is not None else result.last_hidden_state[:, 0]
        return F.normalize(value.float(), dim=1).cpu()


@torch.no_grad()
def probe_raw_future_causality(vae, frames, cuts, *, atol=1e-5, rtol=1e-4,
                               required_context_frames=None):
    """Poison RAW future pixels and compare all admissible past latent outputs.

    Prefix-only encoding must also match encoding that prefix inside a longer
    clip. Latent-only poisoning would miss full-clip encoder normalization.
    """
    assert_frozen(vae)
    if not isinstance(frames, torch.Tensor) or frames.ndim != 5 or frames.shape[2] != 3:
        raise GateError("raw causal probe requires (N,T,3,H,W)")
    if not bool(torch.isfinite(frames).all()) or bool(((frames < 0) | (frames > 1)).any()):
        raise GateError("raw probe must be finite RGB in [0,1]")
    stride, n = int(vae.t_stride), frames.shape[1]
    if stride < 1 or (n-1) % stride or not cuts:
        raise GateError("probe length/cuts must use the exact 1+stride*k raw frame grid; no trimming")
    if any(cut < 1 or cut >= n or (cut-1) % stride for cut in cuts):
        raise GateError("every cut must be a valid observed prefix with a nonempty future")
    if required_context_frames is not None and (required_context_frames not in cuts
                                                or n < required_context_frames+stride):
        raise GateError("causal gate must cover the entire learned observed context plus an unseen stride block")
    input_hash = tensor_sha256(frames)
    full = vae.encode(frames)
    if full.shape[1] != 1+(n-1)//stride or not bool(torch.isfinite(full).all()):
        raise GateError("encoder silently dropped frames or produced invalid latents")
    rows = []
    for cut in cuts:
        expected = 1+(cut-1)//stride
        prefix = vae.encode(frames[:, :cut])
        if prefix.shape[1] != expected:
            raise GateError("prefix encoder output has the wrong raw-frame mapping")
        for poison in ("zero", "invert", "seeded_noise"):
            altered = frames.clone()
            suffix = altered[:, cut:]
            if poison == "zero":
                suffix.zero_()
            elif poison == "invert":
                suffix.copy_(1-suffix)
            else:
                suffix.copy_(torch.rand(suffix.shape, generator=torch.Generator().manual_seed(19013),
                                        dtype=suffix.dtype).to(suffix.device))
            poisoned = vae.encode(altered)[:, :expected]
            if poisoned.shape != prefix.shape or full[:, :expected].shape != prefix.shape:
                raise GateError("past latent shape depends on future clip length")
            scale = max(float(prefix.abs().max()), 1.0)
            tolerance = atol+rtol*scale
            poison_delta = float((poisoned-prefix).abs().max())
            context_delta = float((full[:, :expected]-prefix).abs().max())
            rows.append({"raw_prefix_frames": cut, "past_latent_steps": expected, "poison": poison,
                         "poison_delta_max": poison_delta, "longer_context_delta_max": context_delta,
                         "tolerance": tolerance,
                         "pass": math.isfinite(poison_delta) and math.isfinite(context_delta)
                                 and poison_delta <= tolerance and context_delta <= tolerance})
    assert_frozen(vae)
    if tensor_sha256(frames) != input_hash:
        raise GateError("encoder mutated the observed raw input")
    return {"pass": all(row["pass"] for row in rows), "rows": rows, "raw_frame_hash": input_hash,
            "raw_frames": n, "t_stride": stride, "atol": atol, "rtol": rtol,
            "required_context_frames": required_context_frames,
            "covers_required_context": required_context_frames is not None,
            "coverage_scope": "tested raw lengths/cuts only; short probes cannot certify untested longer clips",
            "target_or_future_observation_passed_to_predictor": False}


@dataclass(frozen=True)
class FairProtocol:
    initialization_seeds: tuple = (0, 1, 2)
    raw_context_frames: int = 33
    raw_hidden_gap: int = 256
    chunk_latents: int = 32
    attention_window_latents: int = 32
    training_updates: int = 1000
    heldout_raw_generation_frames: int = 7200
    required_arms: tuple = ("wave", "generic_ssm", "bounded_sdpa", "factorized_attention", "sink_reference")
    required_controls: tuple = ("observed_reference", "observed_rgb_latch", "last_context", "privileged_oracle")

    def validate(self, stride):
        integers = (stride, self.raw_context_frames, self.raw_hidden_gap, self.chunk_latents,
                    self.attention_window_latents, self.training_updates, self.heldout_raw_generation_frames)
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in integers):
            raise GateError("stride, context, horizons and budgets must be positive integers")
        if self.initialization_seeds != (0, 1, 2) or self.raw_hidden_gap < 256:
            raise GateError("all learned arms require initialization seeds 0/1/2 and >=256 RAW hidden frames")
        if (self.raw_context_frames-1) % stride or self.raw_hidden_gap % stride:
            raise GateError("raw context/gap must fit the same VAE grid without shortening or padding time")
        if min(self.chunk_latents, self.attention_window_latents, self.training_updates) < 1:
            raise GateError("invalid common budgets")
        mandatory = {"wave", "generic_ssm", "bounded_sdpa", "factorized_attention", "sink_reference"}
        if not mandatory <= set(self.required_arms) or len(set(self.required_arms)) != len(self.required_arms):
            raise GateError("factorized attention and sink/reference shipping comparators are mandatory")
        return self

    def mapping(self, stride):
        self.validate(stride)
        observed = self.raw_context_frames+self.raw_hidden_gap
        return {**asdict(self), "t_stride": stride, "observed_raw_frames_per_arm": observed,
                "observed_latent_steps_per_arm": 1+(observed-1)//stride,
                "cue_last_raw_index": self.raw_context_frames-1,
                "hidden_raw_interval": [self.raw_context_frames, observed],
                "first_unseen_raw_index": observed,
                "next_latent_decoded_raw_interval": [observed, observed+stride],
                "attention_raw_retention_frames": self.attention_window_latents*stride,
                "attention_training_gap_capped": False, "same_target_examples_loss_and_updates": True}

    def assert_run_rows(self, rows):
        self.validate(1)
        wanted = {(arm, seed) for arm in self.required_arms for seed in self.initialization_seeds}
        actual = {(row["arm"], row["initialization_seed"]) for row in rows}
        if actual != wanted or len(rows) != len(wanted):
            raise GateError("comparison requires every registered arm x all three initialization seeds, exactly once")
        for row in rows:
            if row.get("completed") is not True or row.get("completed_updates") != self.training_updates:
                raise GateError("declared plans or incomplete training runs cannot satisfy the completed-run gate")
            sources = row.get("heldout_source_ids")
            if (not isinstance(sources, (list, tuple)) or not sources
                    or any(not isinstance(source, str) or not source for source in sources)
                    or len(set(sources)) != len(sources)):
                raise GateError("completed runs need nonempty distinct held-out source identities")
        shared = ("vae_fingerprint", "dataset_fingerprint", "raw_context_frames", "raw_hidden_gap",
                  "raw_observed_frames", "training_updates", "loss_id", "heldout_source_ids", "t_stride",
                  "target_raw_interval", "decoder_lookahead_latents", "heldout_raw_generation_frames")
        for key in shared:
            if any(row.get(key) != rows[0].get(key) for row in rows) or rows[0].get(key) is None:
                raise GateError(f"arms differ or lack a common budget/source identity: {key}")
        if (rows[0]["raw_context_frames"] != self.raw_context_frames or rows[0]["raw_hidden_gap"] != self.raw_hidden_gap
                or rows[0]["raw_observed_frames"] != self.raw_context_frames+self.raw_hidden_gap
                or rows[0]["training_updates"] != self.training_updates
                or rows[0]["heldout_raw_generation_frames"] != self.heldout_raw_generation_frames):
            raise GateError("actual runs changed the registered raw horizon or update budget")
        expected = self.mapping(rows[0]["t_stride"])["next_latent_decoded_raw_interval"]
        if rows[0]["target_raw_interval"] != expected:
            raise GateError("target pixels do not correspond to the first unobserved VAE block")


@torch.no_grad()
def decoded_health_report(context, generated, *, fps=24, stride=1, features=None, patience=24,
                          fixture_features=False):
    """Native-pixel health against a fixed external clean context, never a later generated window.

    Optional pretrained frozen image features add a fixed anchor drift screen.
    These heuristic screens never establish semantic quality or blind preference.
    """
    if (context.ndim != 5 or generated.ndim != 5 or min(context.shape[:2]) < 1
            or context.shape[2] != 3 or context.shape[0] != generated.shape[0]
            or context.shape[2:] != generated.shape[2:]):
        raise GateError("decoded context and generated native pixel grids must match")
    if generated.shape[1] < 1 or fps <= 0 or stride < 1:
        raise GateError("nonempty generation and positive fps/stride required")
    context, generated = context.detach().cpu().float(), generated.detach().cpu().float()
    monitor = HealthMonitor(patience=patience).calibrate(context)
    for frame in generated.unbind(1):
        monitor.update(frame)
    report = {"native_resolution_hw": list(generated.shape[-2:]), "decoded_frames": generated.shape[1],
              "fixed_context_sha256": tensor_sha256(context), "anchor_source": "separate_observed_clean_context",
              "pixel_screen": monitor.config(), "first_collapse": {},
              "frozen_feature_screen": {"status": "not_run"}, "semantic_quality": "unverified",
              "independent_native_pixels_and_blind_human_review_required": True}
    for flag, frame in monitor.first.items():
        report["first_collapse"][flag] = {"decoded_frame": frame, "seconds": frame/fps,
            "latent_step": frame//stride, "sample": monitor.first_sample[flag]}
    if features is not None:
        assert_frozen(features)
        if not fixture_features and not isinstance(features, FrozenImageFeatures):
            raise GateError("frozen image feature screen requires pinned pretrained provenance; no pixel/random fallback")
        n, t = context.shape[:2]
        anchor = features(context.flatten(0, 1)).reshape(n, t, -1).mean(1)
        anchor = F.normalize(anchor.float(), dim=1)
        runs = torch.zeros(n, dtype=torch.long)
        first = None
        similarities = []
        for i, frame in enumerate(generated.unbind(1)):
            valid = torch.isfinite(frame.flatten(1)).all(1)
            similarity = torch.full((n,), -1.0)
            if bool(valid.any()):
                similarity[valid] = (F.normalize(features(frame[valid]).float(), dim=1)*anchor[valid]).sum(1)
            similarities.append(similarity.tolist())
            runs = torch.where(similarity < .9, runs+1, torch.zeros_like(runs))
            hit = (runs >= patience).nonzero().flatten()
            if len(hit) and first is None:
                first = {"decoded_frame": i-patience+1, "seconds": (i-patience+1)/fps,
                         "latent_step": (i-patience+1)//stride, "sample": int(hit[0])}
        report["frozen_feature_screen"] = {"status": "fixture_only_not_pretrained" if fixture_features else "run",
                                           "identity": getattr(features, "proof", {"pretrained": False}), "threshold": .9,
                                           "patience": patience, "reference": "fixed_clean_context",
                                           "cosine_per_frame_sample": similarities, "first_drift": first}
    return report


def pixel_metrics(prediction, reference):
    if (prediction.shape != reference.shape or not bool(torch.isfinite(prediction).all())
            or not bool(torch.isfinite(reference).all())):
        raise GateError("pixel metric inputs must have matching finite native grids")
    mse = float((prediction.float()-reference.float()).square().mean())
    return {"mse": mse, "psnr_db": None if mse == 0 else -10*math.log10(mse),
            "exact_match": mse == 0, "semantic_quality": "unverified"}


@torch.no_grad()
def probe_decoder_chunking(vae, channels, height, width, *, chunk_sizes=(7, 13),
                           generated_latents=28, max_rf_latents=320, atol=5e-4, rtol=1e-4):
    """Actual-weight RF, lookahead, seam and resume gate at bounded spatial sizes.

    This tests decoder plumbing with synthetic latents; it is neither learned
    generation nor native-resolution quality evidence. The separate native RGB
    health/independent footage gates remain required.
    """
    assert_frozen(vae)
    try:
        back, forward = measure_temporal_rf(vae, channels, height, width, max_n=max_rf_latents,
                                           probe_hw=8)
    except SystemExit as error:
        raise GateError(f"pretrained decoder receptive-field probe rejected: {error}") from error
    history, lookahead = history_with_margin(back), forward+1
    h, w = min(height, 4), min(width, 4)
    generator = torch.Generator().manual_seed(19013)
    context = torch.randn(1, history, channels, h, w, generator=generator)*.5
    generated = torch.randn(1, generated_latents, channels, h, w, generator=generator)*.5
    full = vae.decode(torch.cat((context, generated), 1)).cpu()
    if not bool(torch.isfinite(full).all()):
        raise GateError("pretrained decoder produced nonfinite pixels")
    reference = full[:, 1+(history-1)*vae.t_stride:]
    rows = []
    for chunk in chunk_sizes:
        if chunk < 1:
            raise GateError("decoder chunk lengths must be positive")
        decoder = StreamDecoder(vae, history, lookahead)
        decoder.seed(context)
        outputs = []
        for i in range(0, generated_latents, chunk):
            value = decoder.push(generated[:, i:i+chunk])
            if value is not None:
                outputs.append(value)
            # Verify real retained history/lookahead survives an interrupted run.
            if i == 0:
                saved = decoder.state_dict()
                decoder = StreamDecoder(vae, history, lookahead)
                decoder.load_state_dict(saved)
        value = decoder.flush()
        if value is not None:
            outputs.append(value)
        actual = torch.cat(outputs, 1)
        if actual.shape != reference.shape:
            raise GateError("chunked decoder lost or duplicated raw frame timeline")
        delta = float((actual-reference).abs().max())
        tolerance = atol+rtol*max(float(reference.abs().max()), 1.0)
        rows.append({"chunk_latents": chunk, "decoded_frames": actual.shape[1], "delta_max": delta,
                     "tolerance": tolerance, "pass": math.isfinite(delta) and delta <= tolerance})
    return {"pass": all(row["pass"] for row in rows), "rf_back_latents": back, "rf_forward_latents": forward,
            "history_latents": history, "lookahead_latents": lookahead,
            "lookahead_raw_frames": lookahead*vae.t_stride,
            "rf_spatial_cap_latents": 8, "seam_spatial_hw_latents": [h, w], "rows": rows,
            "resume_tested": True, "synthetic_latent_probe": True,
            "native_resolution_quality_evidence": False, "max_rf_latents": max_rf_latents}
