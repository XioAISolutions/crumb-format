"""Native learned-memory adapters behind the frozen observed-only frontend.

Feature extraction is causal and nondifferentiable. The heads receive only
signed, fixed-world innovations. They cannot read the RGB latch, appearances,
camera metadata, targets, or labels. Decode uses the recipient's observed
reference and observed-derived forecast transform; it never clamps RGB/signs.

Wave/SSM retain native recurrent state through all true 32-frame chunks.
Attention consumes the same sequence but retains only its last 32 innovations;
a cue 256 frames earlier is outside that baseline's context. This is an
integration scaffold, not learned-recall or architectural-superiority evidence.

Chunk ingest exposes only the final prediction, after all its input observations.
The native wave temporal FFT's dense intermediate outputs can amplify tiny
roundoff on blank tokens through RMSNorm (measured sparse-prefix counterexample).
Public per-frame prefix predictions therefore use the actual native frame-step
recurrence, which never receives later frames, rather than those dense FFT rows.
The wave chunk adapter is restricted to one layer: those numerical intermediate
artifacts can feed later layers and break full-stack chunk/frame-state parity.
"""
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Optional
import importlib
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .observation import ObservationConfig, ObservationMemory
except ImportError:
    from observation import ObservationConfig, ObservationMemory


NATIVE_ROOT = Path(__file__).resolve().parent.parent
if str(NATIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(NATIVE_ROOT))


def _native_module(name):
    module = importlib.import_module(name)
    if Path(module.__file__).resolve() != NATIVE_ROOT / (name + ".py"):
        raise RuntimeError(f"{name} is shadowed; use a fresh process with the canonical native source")
    return module


VideoPredictor = _native_module("wfvideo").VideoPredictor


@dataclass(frozen=True)
class FeatureRecord:
    features: torch.Tensor                    # [N,T,3,H,W], first-view world coordinates
    reference_canonical: torch.Tensor         # [N,3,H,W], forecast view, latch disabled
    known: torch.Tensor                       # [N,1,H,W], observed reference support
    forecast_positions: torch.Tensor          # [N,2], (x,y) relative to first view
    gains: torch.Tensor                       # observed = gain * canonical + offset
    offsets: torch.Tensor
    accepted: torch.Tensor                    # [N,T], actual frontend decisions
    coverage: torch.Tensor                    # [N,T], alignment reference overlap
    frame_offset: int
    total_observed_frames: int

    @property
    def unknown(self):
        return ~self.known

    def to(self, device):
        return replace(self, **{f.name: getattr(self, f.name).to(device)
                               for f in fields(self) if isinstance(getattr(self, f.name), torch.Tensor)})


class FeatureExtractor:
    """A batch of causal frontend streams, retained across extract() calls.

    Retained frontend RGB latches are never copied into FeatureRecord or used
    by decode_prediction. Their writes supply only the current innovation.
    """

    def __init__(self, config: Optional[ObservationConfig] = None):
        self.config = config or ObservationConfig()
        self._streams = None
        self._shape = None
        self.observed_frames = 0

    @torch.no_grad()
    def extract(self, frames: torch.Tensor) -> FeatureRecord:
        if not isinstance(frames, torch.Tensor) or frames.ndim != 5 or min(frames.shape[:2]) < 1:
            raise ValueError("observed frames must have shape (N,T,3,H,W), N,T >= 1")
        n, t, c, h, w = frames.shape
        if c != 3 or min(h, w) < 3 or not frames.is_floating_point() or not bool(torch.isfinite(frames).all()):
            raise ValueError("finite floating RGB observations with H,W >= 3 required")
        if self._shape is not None and (n, c, h, w) != self._shape:
            raise ValueError("batch/frame shape cannot change in a frontend stream")
        if self._streams is None:
            self._shape = (n, c, h, w)
            self._streams = [ObservationMemory(self.config) for _ in range(n)]
        traces = [[stream.step(frame) for frame in sequence]
                  for stream, sequence in zip(self._streams, frames)]
        forecasts = [stream.forecast(use_latch=False, observed_units=False) for stream in self._streams]
        offset = self.observed_frames
        self.observed_frames += t
        return FeatureRecord(
            features=torch.stack([torch.stack([r.canonical_innovation for r in trace]) for trace in traces]),
            reference_canonical=torch.stack([f["rgb"] for f in forecasts]),
            known=torch.stack([f["known"] for f in forecasts]),
            forecast_positions=torch.tensor([f["position"] for f in forecasts], dtype=torch.float32),
            gains=torch.tensor([f["gain"] for f in forecasts], dtype=torch.float32),
            offsets=torch.tensor([f["offset"] for f in forecasts], dtype=torch.float32),
            accepted=torch.tensor([[r.accepted for r in trace] for trace in traces]),
            coverage=torch.tensor([[r.coverage for r in trace] for trace in traces]),
            frame_offset=offset, total_observed_frames=self.observed_frames,
        )

    def state_dict(self):
        return {"observed_frames": self.observed_frames, "shape": self._shape,
                "streams": None if self._streams is None else [s.state_dict() for s in self._streams]}


@torch.no_grad()
def extract_features(frames: torch.Tensor, config: Optional[ObservationConfig] = None,
                     *, chunk_size: int = 32) -> FeatureRecord:
    """Replay observations in true chunks; no clip mean, targets, or future pooling."""
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if not isinstance(frames, torch.Tensor) or frames.ndim != 5 or min(frames.shape[:2]) < 1:
        raise ValueError("observed frames must have shape (N,T,3,H,W), N,T >= 1")
    extractor = FeatureExtractor(config)
    records = [extractor.extract(frames[:, i:i+chunk_size]) for i in range(0, frames.shape[1], chunk_size)]
    return replace(records[-1], features=torch.cat([r.features for r in records], 1),
                   accepted=torch.cat([r.accepted for r in records], 1),
                   coverage=torch.cat([r.coverage for r in records], 1), frame_offset=0)


@dataclass(frozen=True)
class HeadState:
    kind: str
    processed_frames: int
    recurrent: Optional[list] = None
    window: Optional[torch.Tensor] = None

    def detach(self):
        return HeadState(self.kind, self.processed_frames,
                         None if self.recurrent is None else [x.detach() for x in self.recurrent],
                         None if self.window is None else self.window.detach())


class MemoryHead(nn.Module):
    """An actual native VideoPredictor readout of signed canonical innovations.

    ingest preserves the recurrent autograd graph across chunks. A trainer
    chooses whether to detach HeadState between optimizer steps; it must not
    replace the registered 256-frame gap with a 32-frame training shortcut.
    Native generic SSM and bounded attention preserve their existing biases and
    spatial embeddings. Wave explicitly uses halflife/dispersion/clean_write.
    Equal dimensions do not equalize compute, parameters, or retained bytes.
    """

    def __init__(self, kind: str, height: int, width: int, *, dim: int = 16,
                 layers: int = 1, heads: int = 2, window: int = 32,
                 hl_min: float = 2.0, hl_max: float = 4096.0):
        super().__init__()
        if kind not in ("wave", "ssm", "attn"):
            raise ValueError("kind must be wave, ssm, or attn")
        if min(height, width, dim, layers, heads, window) < 1 or dim % heads:
            raise ValueError("positive dimensions and dim divisible by heads required")
        if kind == "wave" and layers != 1:
            raise ValueError("wave chunk adapter supports layers=1; multilayer FFT/frame parity is not established")
        if kind == "ssm":
            _native_module("ssm_lite")
        self.kind, self.height, self.width, self.window = kind, height, width, window
        kwargs = dict(causal=True, residual=False, time_pos="none")
        if kind == "wave":
            kwargs.update(kernel_version="dispersion", linear_pad=True, pole_param="halflife",
                          hl_min=hl_min, hl_max=hl_max, clean_write=True)
        self.native = VideoPredictor(dim, layers, heads, window, height, width, kind, **kwargs)

    @contextmanager
    def _true_length(self, length):
        """Native constructors use fixed T; a partial chunk uses its real length.

        This changes this instance's runtime sizes only, restoring them after
        forward. It never pads fictional time, changes native source/defaults,
        or replaces the native operator/parameters.
        """
        saved = [(self.native, self.native.T), (self.native.posemb, self.native.posemb.T)]
        masks = []
        for block in self.native.blocks:
            saved.append((block.mix, block.mix.T))
            if self.kind == "attn":
                mask = block.mix.attn_mask
                masks.append((block.mix, mask))
                tokens = length*self.height*self.width
                block.mix.attn_mask = mask[:tokens, :tokens]
        try:
            for module, _ in saved:
                module.T = length
            yield
        finally:
            for module, old_length in saved:
                module.T = old_length
            for module, mask in masks:
                module.attn_mask = mask

    def _features(self, features):
        if not isinstance(features, torch.Tensor) or features.ndim != 5 or min(features.shape[:2]) < 1:
            raise ValueError("head input must be (N,T,3,H,W), N,T >= 1")
        if tuple(features.shape[2:]) != (3, self.height, self.width):
            raise ValueError("feature grid does not match this head")
        if not features.is_floating_point() or not bool(torch.isfinite(features).all()):
            raise ValueError("signed features must be finite floating point")
        parameter = next(self.parameters())
        if features.device != parameter.device or features.dtype != parameter.dtype:
            raise ValueError("features must match the head's device and floating dtype")

    def ingest(self, features: torch.Tensor, state: Optional[HeadState] = None):
        self._features(features)
        # Native checkpoint recomputation reads the restored instance T. Full
        # window chunks are safe; a partial differentiable chunk must use the
        # normal native backward path (grad_ckpt=False), never fail much later.
        if (self.kind != "attn" and self.native.grad_ckpt and torch.is_grad_enabled()
                and features.shape[1] % self.window):
            raise ValueError("partial native chunks require grad_ckpt=False for backward")
        if state is not None and state.kind != self.kind:
            raise ValueError("state belongs to a different operator")
        processed = 0 if state is None else state.processed_frames
        if self.kind == "attn":
            window = None if state is None else state.window
            for i in range(0, features.shape[1], self.window):
                part = features[:, i:i+self.window]
                joined = part if window is None else torch.cat((window, part), 1)
                window = joined[:, -self.window:].clone()
                processed += part.shape[1]
            with self._true_length(window.shape[1]):
                predicted = self.native(window)
            return predicted, HeadState(self.kind, processed, window=window)
        recurrent = [None]*len(self.native.blocks) if state is None else state.recurrent
        predicted = None
        for i in range(0, features.shape[1], self.window):
            part = features[:, i:i+self.window]
            with self._true_length(part.shape[1]):
                predicted, recurrent = self.native(part, states=recurrent)
            processed += part.shape[1]
        return predicted, HeadState(self.kind, processed, recurrent=recurrent)

    def forward(self, features: torch.Tensor):
        return self.ingest(features)[0]

    def step(self, feature: torch.Tensor, state: Optional[HeadState] = None):
        """Strict one-observation native recurrence; no temporal FFT prefix rows."""
        if not isinstance(feature, torch.Tensor) or feature.ndim != 4:
            raise ValueError("one feature frame must have shape (N,3,H,W)")
        self._features(feature[:, None])
        if self.kind == "attn":
            return self.ingest(feature[:, None], state)
        if state is not None and state.kind != self.kind:
            raise ValueError("state belongs to a different operator")
        processed = 0 if state is None else state.processed_frames
        recurrent = (self.native.stream_init(feature.shape[0], feature.device) if state is None
                     else list(state.recurrent))
        predicted, recurrent = self.native.stream_step(feature, recurrent, processed)
        return predicted, HeadState(self.kind, processed+1, recurrent=recurrent)

    def prefix_predictions(self, features: torch.Tensor, state: Optional[HeadState] = None):
        """Per-frame causal predictions; retains autograd for optional dense training."""
        self._features(features)
        predictions = []
        for feature in features.unbind(1):
            predicted, state = self.step(feature, state)
            predictions.append(predicted)
        return torch.stack(predictions, 1), state

    @torch.no_grad()
    def predict(self, features: torch.Tensor, state: Optional[HeadState] = None):
        """Inference API; returns prediction and bounded state without an autograd graph."""
        return self.ingest(features, state)


Head = MemoryHead


@dataclass(frozen=True)
class DecodedPrediction:
    rgb: torch.Tensor
    canonical_event: torch.Tensor
    view_event: torch.Tensor
    reference_canonical: torch.Tensor
    known: torch.Tensor

    @property
    def unknown(self):
        return ~self.known


def decode_prediction(canonical_event: torch.Tensor, record: FeatureRecord) -> DecodedPrediction:
    """Add the learned event to this recipient's own observed reference.

    Signed canonical event pixels are sampled into the forecast view using only
    the frontend's observed-derived position. Latest accepted gain/offset then
    restore observed units. Zero-event decode is the reference-only baseline.
    """
    if not isinstance(record, FeatureRecord):
        raise ValueError("decoder requires observed-only FeatureRecord context")
    if not isinstance(canonical_event, torch.Tensor) or canonical_event.shape != record.reference_canonical.shape:
        raise ValueError("canonical event must have shape (N,3,H,W) matching the observed reference")
    if not canonical_event.is_floating_point() or not bool(torch.isfinite(canonical_event).all()):
        raise ValueError("canonical event must be finite floating point")
    if canonical_event.device != record.reference_canonical.device:
        raise ValueError("event and observed context must share a device")
    n, _, h, w = canonical_event.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=canonical_event.device),
                            torch.arange(w, device=canonical_event.device), indexing="ij")
    x = xx[None] + record.forecast_positions[:, 0, None, None]
    y = yy[None] + record.forecast_positions[:, 1, None, None]
    grid = torch.stack((2*x/(w-1)-1, 2*y/(h-1)-1), -1).to(canonical_event.dtype)
    view = F.grid_sample(canonical_event, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    rgb = (record.reference_canonical + view) * record.gains[:, None, None, None] + record.offsets[:, None, None, None]
    return DecodedPrediction(rgb, canonical_event, view, record.reference_canonical, record.known)
