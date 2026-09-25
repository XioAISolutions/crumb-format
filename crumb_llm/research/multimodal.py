"""Multi-Modal Wave Fields for unified text, image, audio, and code processing.

This module extends wave propagation to handle multiple modalities in a unified
wave field, enabling cross-modal reasoning and generation.

Key innovations:
- Separate wave fields per modality with cross-modal interference
- Modality-specific scatter-gather (vision patches, audio spectrograms, code AST)
- Cross-modal attention via wave interference patterns
- Unified tokenization and embedding space
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..model import WaveFieldConfig, WaveFieldLM
from ..layers import WaveFieldBlock, WaveFieldBlockConfig, RMSNorm, SwiGLUFFN
from ..scatter_gather import scatter_linear, gather_linear


ModalityType = Literal["text", "image", "audio", "code"]


@dataclass
class ModalityConfig:
    """Configuration for a single modality."""
    name: ModalityType
    vocab_size: int
    patch_size: int = 1  # For vision: patch size, for audio: frame size
    encoder_dim: int = 512
    use_pretrained: bool = False
    pretrained_path: Optional[str] = None


class ModalityEncoder(nn.Module):
    """Encode a specific modality into the unified wave field space.
    
    Each modality has its own encoder that projects raw inputs into the
    shared embedding dimension, then scatters onto its own wave field.
    """
    
    def __init__(self, config: ModalityConfig, target_dim: int):
        super().__init__()
        self.config = config
        self.target_dim = target_dim
        
        if config.name == "text":
            self.encoder = nn.Embedding(config.vocab_size, target_dim)
        elif config.name == "image":
            # Vision encoder: patch embedding + position encoding
            self.patch_embed = nn.Conv2d(
                3, config.encoder_dim, 
                kernel_size=config.patch_size, 
                stride=config.patch_size
            )
            self.proj = nn.Linear(config.encoder_dim, target_dim)
        elif config.name == "audio":
            # Audio encoder: spectrogram CNN + projection
            self.spec_conv = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((1, config.encoder_dim)),
            )
            self.proj = nn.Linear(config.encoder_dim, target_dim)
        elif config.name == "code":
            # Code encoder: token + AST structure embedding
            self.token_embed = nn.Embedding(config.vocab_size, target_dim)
            self.ast_embed = nn.Embedding(100, target_dim)  # AST node types
        
        self.norm = RMSNorm(target_dim)
    
    def forward(self, x: Tensor, ast_types: Optional[Tensor] = None) -> Tensor:
        """Encode modality input to unified embedding space.
        
        Args:
            x: Input tensor (shape depends on modality)
            ast_types: Optional AST node types for code modality
            
        Returns:
            Encoded tensor [B, N, D] where N depends on modality
        """
        if self.config.name == "text":
            # x: [B, N] token ids
            return self.norm(self.encoder(x))
        
        elif self.config.name == "image":
            # x: [B, C, H, W] image
            B = x.shape[0]
            patches = self.patch_embed(x)  # [B, encoder_dim, H', W']
            patches = patches.flatten(2).transpose(1, 2)  # [B, N_patches, encoder_dim]
            return self.norm(self.proj(patches))
        
        elif self.config.name == "audio":
            # x: [B, 1, T, F] spectrogram
            B = x.shape[0]
            features = self.spec_conv(x)  # [B, 128, 1, encoder_dim]
            features = features.squeeze(2).transpose(1, 2)  # [B, encoder_dim, 128]
            features = features.mean(dim=2)  # [B, encoder_dim]
            return self.norm(self.proj(features.unsqueeze(1)))
        
        elif self.config.name == "code":
            # x: [B, N] token ids
            token_emb = self.token_embed(x)
            if ast_types is not None:
                ast_emb = self.ast_embed(ast_types)
                return self.norm(token_emb + ast_emb)
            return self.norm(token_emb)
        
        raise ValueError(f"Unknown modality: {self.config.name}")


class CrossModalInterference(nn.Module):
    """Enable wave interference between different modality fields.
    
    This allows information to flow between modalities through wave
    interference patterns, enabling cross-modal reasoning.
    """
    
    def __init__(self, n_modalities: int, dim: int, field_size: int):
        super().__init__()
        self.n_modalities = n_modalities
        self.dim = dim
        self.field_size = field_size
        
        # Learnable coupling coefficients between modalities
        self.coupling = nn.Parameter(torch.eye(n_modalities) * 0.1)
        
        # Phase alignment for each modality pair
        self.phase_align = nn.Parameter(torch.zeros(n_modalities, n_modalities))
    
    def forward(self, fields: list[Tensor]) -> list[Tensor]:
        """Apply cross-modal interference to wave fields.
        
        Args:
            fields: List of [B, H, F, D] field tensors, one per modality
            
        Returns:
            List of interfered fields with same shapes
        """
        if len(fields) != self.n_modalities:
            raise ValueError(f"Expected {self.n_modalities} fields, got {len(fields)}")
        
        # Convert to frequency domain for interference
        specs = [torch.fft.rfft(f, dim=2) for f in fields]
        
        # Apply coupling and phase alignment
        interfered_specs = []
        for i in range(self.n_modalities):
            interfered = torch.zeros_like(specs[i])
            for j in range(self.n_modalities):
                # Coupling strength
                coupling_ij = self.coupling[i, j].abs()
                # Phase alignment
                phase_shift = torch.polar(
                    torch.ones_like(specs[j].real),
                    torch.full_like(specs[j].real, self.phase_align[i, j])
                )
                interfered = interfered + coupling_ij * specs[j] * phase_shift
            interfered_specs.append(interfered)
        
        # Convert back to spatial domain
        return [torch.fft.irfft(s, n=self.field_size, dim=2) for s in interfered_specs]


@dataclass
class MultiModalWaveFieldConfig:
    """Configuration for multi-modal wave field model."""
    base_config: WaveFieldConfig
    modalities: list[ModalityConfig]
    cross_modal_layers: int = 2  # Number of cross-modal interference layers
    fusion_strategy: Literal["early", "late", "hierarchical"] = "hierarchical"


class MultiModalWaveField(nn.Module):
    """Multi-modal wave field language model.
    
    Extends WaveFieldLM to handle multiple modalities (text, images, audio, code)
    in a unified wave field architecture. Each modality has its own wave field
    that can interfere with others, enabling cross-modal reasoning.
    
    Architecture:
    1. Modality-specific encoders project inputs to shared embedding space
    2. Each modality scatters onto its own wave field
    3. Cross-modal interference layers allow information exchange
    4. Unified wave field blocks process all modalities
    5. Modality-specific decoders generate outputs
    """
    
    def __init__(self, config: MultiModalWaveFieldConfig):
        super().__init__()
        self.config = config
        self.base_dim = config.base_config.dim
        
        # Modality encoders
        self.encoders = nn.ModuleDict({
            mod.name: ModalityEncoder(mod, self.base_dim)
            for mod in config.modalities
        })
        
        # Cross-modal interference layers
        self.cross_modal = nn.ModuleList([
            CrossModalInterference(
                len(config.modalities),
                self.base_dim,
                config.base_config.field_size
            )
            for _ in range(config.cross_modal_layers)
        ])
        
        # Unified wave field blocks
        block_cfg = WaveFieldBlockConfig(
            dim=self.base_dim,
            n_heads=config.base_config.n_heads,
            field_size=config.base_config.field_size,
            ffn_mult=config.base_config.ffn_mult,
            causal=config.base_config.causal,
            kernel_mode=config.base_config.kernel_mode,
            dropout=config.base_config.dropout,
        )
        self.blocks = nn.ModuleList([
            WaveFieldBlock(block_cfg)
            for _ in range(config.base_config.n_layers)
        ])
        
        # Modality-specific output heads
        self.output_heads = nn.ModuleDict({
            mod.name: nn.Linear(self.base_dim, mod.vocab_size, bias=False)
            for mod in config.modalities
        })
        
        self.norm_out = RMSNorm(self.base_dim)
    
    def encode_modality(
        self,
        modality: ModalityType,
        x: Tensor,
        **kwargs
    ) -> Tensor:
        """Encode input from a specific modality.
        
        Args:
            modality: Type of modality
            x: Input tensor (shape depends on modality)
            **kwargs: Additional modality-specific arguments
            
        Returns:
            Encoded tensor [B, N, D]
        """
        if modality not in self.encoders:
            raise ValueError(f"Unknown modality: {modality}")
        return self.encoders[modality](x, **kwargs)
    
    def forward(
        self,
        inputs: dict[ModalityType, Tensor],
        targets: Optional[dict[ModalityType, Tensor]] = None,
        **kwargs
    ) -> dict:
        """Forward pass with multi-modal inputs.
        
        Args:
            inputs: Dict mapping modality names to input tensors
            targets: Optional dict of target tensors for loss computation
            **kwargs: Additional arguments (e.g., ast_types for code)
            
        Returns:
            Dict with 'logits' (dict per modality) and optional 'loss'
        """
        # Encode each modality
        encoded = {}
        for modality, x in inputs.items():
            mod_kwargs = {k: v for k, v in kwargs.items() if k.startswith(f"{modality}_")}
            encoded[modality] = self.encode_modality(modality, x, **mod_kwargs)
        
        # Scatter each modality onto its own field
        B = next(iter(encoded.values())).shape[0]
        H = self.config.base_config.n_heads
        F = self.config.base_config.field_size
        d_head = self.base_dim // H
        
        fields = []
        for modality in sorted(encoded.keys()):  # Consistent ordering
            x = encoded[modality]
            N = x.shape[1]
            # Reshape to per-head: [B, N, D] -> [B, H, N, d]
            x_heads = x.view(B, N, H, d_head).transpose(1, 2)
            # Scatter to field
            field = scatter_linear(x_heads, F)
            fields.append(field)
        
        # Apply cross-modal interference
        for interference_layer in self.cross_modal:
            fields = interference_layer(fields)
        
        # Merge fields (average for now, could be learned)
        merged_field = torch.stack(fields).mean(dim=0)  # [B, H, F, d]
        
        # Gather back to tokens (use max sequence length)
        max_N = max(x.shape[1] for x in encoded.values())
        x = gather_linear(merged_field, max_N)  # [B, H, N, d]
        x = x.transpose(1, 2).reshape(B, max_N, self.base_dim)
        
        # Process through wave field blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm_out(x)
        
        # Generate outputs for each modality
        logits = {}
        for modality in inputs.keys():
            N_mod = inputs[modality].shape[1] if inputs[modality].dim() > 1 else 1
            x_mod = x[:, :N_mod, :]
            logits[modality] = self.output_heads[modality](x_mod)
        
        result = {"logits": logits}
        
        # Compute loss if targets provided
        if targets is not None:
            total_loss = 0.0
            for modality, target in targets.items():
                if modality in logits:
                    loss = F.cross_entropy(
                        logits[modality].reshape(-1, logits[modality].size(-1)),
                        target.reshape(-1),
                        ignore_index=-100
                    )
                    total_loss += loss
            result["loss"] = total_loss / len(targets)
        
        return result
    
    @torch.no_grad()
    def generate_multimodal(
        self,
        inputs: dict[ModalityType, Tensor],
        target_modality: ModalityType,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Tensor:
        """Generate output in target modality conditioned on inputs.
        
        Args:
            inputs: Dict of input tensors from various modalities
            target_modality: Modality to generate
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            
        Returns:
            Generated tokens [B, N]
        """
        # Start with empty target or provided prefix
        if target_modality not in inputs:
            # Start with BOS token (assume 0)
            B = next(iter(inputs.values())).shape[0]
            device = next(iter(inputs.values())).device
            target_ids = torch.zeros((B, 1), dtype=torch.long, device=device)
        else:
            target_ids = inputs[target_modality]
        
        for _ in range(max_new_tokens):
            # Add current target to inputs
            current_inputs = {**inputs, target_modality: target_ids}
            
            # Forward pass
            out = self(current_inputs)
            logits = out["logits"][target_modality][:, -1, :] / max(temperature, 1e-6)
            
            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                logits = torch.where(
                    logits < v[..., [-1]],
                    torch.full_like(logits, float("-inf")),
                    logits
                )
            
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            target_ids = torch.cat([target_ids, next_id], dim=1)
        
        return target_ids


# Convenience functions for common multi-modal tasks

def create_vision_language_model(
    vocab_size: int = 50000,
    image_size: int = 224,
    patch_size: int = 16,
    dim: int = 512,
    n_layers: int = 12,
    n_heads: int = 8,
) -> MultiModalWaveField:
    """Create a vision-language model for image captioning and VQA."""
    base_config = WaveFieldConfig(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        field_size=512,
    )
    
    modalities = [
        ModalityConfig(name="text", vocab_size=vocab_size),
        ModalityConfig(
            name="image",
            vocab_size=0,  # Not used for images
            patch_size=patch_size,
            encoder_dim=dim,
        ),
    ]
    
    config = MultiModalWaveFieldConfig(
        base_config=base_config,
        modalities=modalities,
        cross_modal_layers=3,
        fusion_strategy="hierarchical",
    )
    
    return MultiModalWaveField(config)


def create_audio_language_model(
    vocab_size: int = 50000,
    dim: int = 512,
    n_layers: int = 12,
    n_heads: int = 8,
) -> MultiModalWaveField:
    """Create an audio-language model for transcription and understanding."""
    base_config = WaveFieldConfig(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        field_size=512,
    )
    
    modalities = [
        ModalityConfig(name="text", vocab_size=vocab_size),
        ModalityConfig(
            name="audio",
            vocab_size=0,
            encoder_dim=dim,
        ),
    ]
    
    config = MultiModalWaveFieldConfig(
        base_config=base_config,
        modalities=modalities,
        cross_modal_layers=2,
        fusion_strategy="early",
    )
    
    return MultiModalWaveField(config)

# Made with Bob
