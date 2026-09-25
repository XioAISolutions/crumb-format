"""Interpretability and Analysis tools for Wave Field LLM.

Tools to understand and visualize wave-based representations:
- Wave pattern visualization (frequency, phase, amplitude)
- Attention-equivalent heatmaps from wave interference
- Neuron activation analysis in wave space
- Causal tracing through wave propagation
- Concept discovery in wave representations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..model import WaveFieldLM, WaveFieldConfig
from ..scatter_gather import scatter_linear, gather_linear


@dataclass
class AnalysisConfig:
    """Configuration for interpretability analysis."""
    num_frequency_bins: int = 64
    num_phase_bins: int = 32
    top_k_concepts: int = 10
    intervention_strength: float = 1.0


class WaveVisualizer:
    """Visualize wave field patterns and dynamics.
    
    Provides tools to visualize:
    - Spatial wave patterns
    - Frequency spectra
    - Phase relationships
    - Amplitude distributions
    """
    
    def __init__(self, model: WaveFieldLM):
        self.model = model
        self.model.eval()
    
    @torch.no_grad()
    def extract_wave_field(
        self,
        input_ids: Tensor,
        layer_idx: int = 0,
        head_idx: int = 0,
    ) -> Tensor:
        """Extract wave field from a specific layer and head.
        
        Args:
            input_ids: [B, N] input tokens
            layer_idx: Which layer to extract from
            head_idx: Which head to extract from
            
        Returns:
            field: [B, F, D] wave field
        """
        # Forward through model up to target layer
        x = self.model.embed(input_ids)
        
        for i, block in enumerate(self.model.blocks):
            if i == layer_idx:
                # Extract field from this block
                B, N, D = x.shape
                H = self.model.cfg.n_heads
                d_head = D // H
                F = self.model.cfg.field_size
                
                # Project and reshape
                h = block.proj_in(block.norm_mix(x))
                h = h.view(B, N, H, d_head).transpose(1, 2)
                
                # Scatter to field
                field = scatter_linear(h, F)  # [B, H, F, d]
                
                # Extract specific head
                return field[:, head_idx, :, :]  # [B, F, d]
            
            x = block(x)
        
        raise ValueError(f"Layer {layer_idx} not found")
    
    def compute_spectrum(self, field: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute frequency spectrum of wave field.
        
        Args:
            field: [B, F, D] wave field
            
        Returns:
            frequencies: [F//2+1] frequency bins
            power: [B, F//2+1, D] power spectrum
        """
        # FFT
        spec = torch.fft.rfft(field, dim=1)  # [B, F//2+1, D]
        power = spec.abs() ** 2
        
        # Frequency bins
        F = field.shape[1]
        frequencies = torch.fft.rfftfreq(F, device=field.device)
        
        return frequencies, power
    
    def compute_phase(self, field: Tensor) -> Tensor:
        """Compute phase of wave field.
        
        Args:
            field: [B, F, D] wave field
            
        Returns:
            phase: [B, F//2+1, D] phase in radians
        """
        spec = torch.fft.rfft(field, dim=1)
        phase = torch.angle(spec)
        return phase
    
    def plot_wave_pattern(
        self,
        field: Tensor,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot spatial wave pattern.
        
        Args:
            field: [F, D] or [B, F, D] wave field
            save_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
            
            if field.dim() == 3:
                field = field[0]  # Take first batch
            
            # Average over feature dimension
            wave = field.mean(dim=-1).cpu().numpy()
            
            plt.figure(figsize=(12, 4))
            plt.plot(wave)
            plt.xlabel("Field Position")
            plt.ylabel("Amplitude")
            plt.title("Wave Field Pattern")
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
            
            plt.close()
        except ImportError:
            print("matplotlib not available for plotting")
    
    def plot_spectrum(
        self,
        frequencies: Tensor,
        power: Tensor,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot frequency spectrum.
        
        Args:
            frequencies: [F//2+1] frequency bins
            power: [F//2+1, D] or [B, F//2+1, D] power spectrum
            save_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
            
            if power.dim() == 3:
                power = power[0]  # Take first batch
            
            # Average over feature dimension
            power_avg = power.mean(dim=-1).cpu().numpy()
            freq_np = frequencies.cpu().numpy()
            
            plt.figure(figsize=(12, 4))
            plt.semilogy(freq_np, power_avg)
            plt.xlabel("Frequency")
            plt.ylabel("Power (log scale)")
            plt.title("Wave Field Spectrum")
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
            
            plt.close()
        except ImportError:
            print("matplotlib not available for plotting")


class AttentionEquivalent:
    """Compute attention-equivalent patterns from wave interference.
    
    Wave fields don't have explicit attention, but we can compute
    which tokens influence which others through wave propagation.
    """
    
    def __init__(self, model: WaveFieldLM):
        self.model = model
        self.model.eval()
    
    @torch.no_grad()
    def compute_influence_matrix(
        self,
        input_ids: Tensor,
        layer_idx: int = 0,
    ) -> Tensor:
        """Compute token-to-token influence matrix.
        
        Args:
            input_ids: [B, N] input tokens
            layer_idx: Which layer to analyze
            
        Returns:
            influence: [B, N, N] influence matrix (like attention weights)
        """
        B, N = input_ids.shape
        
        # Get baseline output
        x = self.model.embed(input_ids)
        for i, block in enumerate(self.model.blocks):
            if i == layer_idx:
                baseline_out = block(x)
                break
            x = block(x)
        
        # Compute influence by perturbing each token
        influence = torch.zeros(B, N, N, device=input_ids.device)
        
        for token_idx in range(N):
            # Perturb this token's embedding
            x_perturbed = self.model.embed(input_ids).clone()
            x_perturbed[:, token_idx, :] = 0  # Zero out token
            
            # Forward through blocks
            for i, block in enumerate(self.model.blocks):
                if i == layer_idx:
                    perturbed_out = block(x_perturbed)
                    break
                x_perturbed = block(x_perturbed)
            
            # Measure change in output
            diff = (baseline_out - perturbed_out).abs().sum(dim=-1)  # [B, N]
            influence[:, :, token_idx] = diff
        
        # Normalize to [0, 1]
        influence = influence / (influence.max(dim=-1, keepdim=True)[0] + 1e-8)
        
        return influence
    
    def plot_attention_heatmap(
        self,
        influence: Tensor,
        tokens: List[str],
        save_path: Optional[str] = None,
    ) -> None:
        """Plot attention-like heatmap.
        
        Args:
            influence: [N, N] influence matrix
            tokens: List of token strings
            save_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            if influence.dim() == 3:
                influence = influence[0]  # Take first batch
            
            influence_np = influence.cpu().numpy()
            
            plt.figure(figsize=(10, 8))
            plt.imshow(influence_np, cmap="viridis", aspect="auto")
            plt.colorbar(label="Influence")
            plt.xlabel("Source Token")
            plt.ylabel("Target Token")
            plt.title("Token Influence Matrix (Attention-Equivalent)")
            
            if len(tokens) <= 20:  # Only show labels for short sequences
                plt.xticks(range(len(tokens)), tokens, rotation=90)
                plt.yticks(range(len(tokens)), tokens)
            
            if save_path:
                plt.savefig(save_path, bbox_inches="tight")
            else:
                plt.show()
            
            plt.close()
        except ImportError:
            print("matplotlib not available for plotting")


class ConceptDiscovery:
    """Discover interpretable concepts in wave representations.
    
    Uses clustering and probing to identify semantic directions
    in the wave field space.
    """
    
    def __init__(self, model: WaveFieldLM):
        self.model = model
        self.model.eval()
        self.concepts: Dict[str, Tensor] = {}
    
    @torch.no_grad()
    def extract_representations(
        self,
        input_ids_list: List[Tensor],
        layer_idx: int = -1,
    ) -> Tensor:
        """Extract representations from a layer.
        
        Args:
            input_ids_list: List of [N] input sequences
            layer_idx: Which layer (-1 for last)
            
        Returns:
            representations: [num_examples, D] representations
        """
        representations = []
        
        for input_ids in input_ids_list:
            input_ids = input_ids.unsqueeze(0)  # Add batch dim
            
            x = self.model.embed(input_ids)
            
            if layer_idx == -1:
                # Use final layer
                for block in self.model.blocks:
                    x = block(x)
                x = self.model.norm_out(x)
            else:
                # Use specific layer
                for i, block in enumerate(self.model.blocks):
                    x = block(x)
                    if i == layer_idx:
                        break
            
            # Pool over sequence
            rep = x.mean(dim=1)  # [1, D]
            representations.append(rep)
        
        return torch.cat(representations, dim=0)  # [num_examples, D]
    
    def discover_concepts(
        self,
        positive_examples: List[Tensor],
        negative_examples: List[Tensor],
        concept_name: str,
        layer_idx: int = -1,
    ) -> Tensor:
        """Discover a concept direction from examples.
        
        Args:
            positive_examples: List of positive example sequences
            negative_examples: List of negative example sequences
            concept_name: Name for this concept
            layer_idx: Which layer to analyze
            
        Returns:
            direction: [D] concept direction vector
        """
        # Extract representations
        pos_reps = self.extract_representations(positive_examples, layer_idx)
        neg_reps = self.extract_representations(negative_examples, layer_idx)
        
        # Compute concept direction (difference of means)
        pos_mean = pos_reps.mean(dim=0)
        neg_mean = neg_reps.mean(dim=0)
        direction = pos_mean - neg_mean
        
        # Normalize
        direction = direction / (direction.norm() + 1e-8)
        
        # Store concept
        self.concepts[concept_name] = direction
        
        return direction
    
    def probe_concept(
        self,
        input_ids: Tensor,
        concept_name: str,
        layer_idx: int = -1,
    ) -> float:
        """Measure how much a sequence exhibits a concept.
        
        Args:
            input_ids: [N] input sequence
            concept_name: Name of concept to probe
            layer_idx: Which layer to analyze
            
        Returns:
            score: Concept activation score
        """
        if concept_name not in self.concepts:
            raise ValueError(f"Concept '{concept_name}' not found")
        
        # Extract representation
        rep = self.extract_representations([input_ids], layer_idx)[0]
        
        # Project onto concept direction
        direction = self.concepts[concept_name]
        score = (rep * direction).sum().item()
        
        return score
    
    def intervene_concept(
        self,
        input_ids: Tensor,
        concept_name: str,
        strength: float = 1.0,
        layer_idx: int = -1,
    ) -> Tensor:
        """Intervene on a concept to modify generation.
        
        Args:
            input_ids: [B, N] input tokens
            concept_name: Concept to intervene on
            strength: Intervention strength (positive or negative)
            layer_idx: Which layer to intervene at
            
        Returns:
            logits: [B, N, V] modified logits
        """
        if concept_name not in self.concepts:
            raise ValueError(f"Concept '{concept_name}' not found")
        
        direction = self.concepts[concept_name]
        
        # Forward with intervention
        x = self.model.embed(input_ids)
        
        for i, block in enumerate(self.model.blocks):
            x = block(x)
            
            # Intervene at target layer
            if i == layer_idx or (layer_idx == -1 and i == len(self.model.blocks) - 1):
                # Add concept direction
                x = x + strength * direction.view(1, 1, -1)
        
        x = self.model.norm_out(x)
        
        if self.model.lm_head is None:
            logits = x @ self.model.embed.weight.t()
        else:
            logits = self.model.lm_head(x)
        
        return logits


class WaveFieldAnalyzer:
    """Complete analysis suite for Wave Field LLM.
    
    Combines visualization, attention analysis, and concept discovery.
    """
    
    def __init__(self, model: WaveFieldLM, config: Optional[AnalysisConfig] = None):
        self.model = model
        self.config = config or AnalysisConfig()
        
        self.visualizer = WaveVisualizer(model)
        self.attention = AttentionEquivalent(model)
        self.concepts = ConceptDiscovery(model)
    
    def analyze_sequence(
        self,
        input_ids: Tensor,
        tokens: Optional[List[str]] = None,
        layer_idx: int = 0,
    ) -> Dict[str, any]:
        """Complete analysis of a sequence.
        
        Args:
            input_ids: [N] input tokens
            tokens: Optional list of token strings
            layer_idx: Which layer to analyze
            
        Returns:
            analysis: Dict with various analysis results
        """
        input_ids = input_ids.unsqueeze(0)  # Add batch dim
        
        # Extract wave field
        field = self.visualizer.extract_wave_field(input_ids, layer_idx)
        
        # Compute spectrum
        frequencies, power = self.visualizer.compute_spectrum(field)
        
        # Compute phase
        phase = self.visualizer.compute_phase(field)
        
        # Compute influence matrix
        influence = self.attention.compute_influence_matrix(input_ids, layer_idx)
        
        analysis = {
            "field": field,
            "frequencies": frequencies,
            "power_spectrum": power,
            "phase": phase,
            "influence_matrix": influence,
        }
        
        return analysis
    
    def compare_sequences(
        self,
        input_ids_1: Tensor,
        input_ids_2: Tensor,
        layer_idx: int = 0,
    ) -> Dict[str, float]:
        """Compare wave representations of two sequences.
        
        Args:
            input_ids_1: [N1] first sequence
            input_ids_2: [N2] second sequence
            layer_idx: Which layer to compare
            
        Returns:
            similarities: Dict of similarity metrics
        """
        # Extract fields
        field1 = self.visualizer.extract_wave_field(
            input_ids_1.unsqueeze(0), layer_idx
        )[0]
        field2 = self.visualizer.extract_wave_field(
            input_ids_2.unsqueeze(0), layer_idx
        )[0]
        
        # Pad to same length
        max_len = max(field1.shape[0], field2.shape[0])
        field1_padded = F.pad(field1, (0, 0, 0, max_len - field1.shape[0]))
        field2_padded = F.pad(field2, (0, 0, 0, max_len - field2.shape[0]))
        
        # Spatial similarity
        spatial_sim = F.cosine_similarity(
            field1_padded.flatten(),
            field2_padded.flatten(),
            dim=0
        ).item()
        
        # Spectral similarity
        _, power1 = self.visualizer.compute_spectrum(field1.unsqueeze(0))
        _, power2 = self.visualizer.compute_spectrum(field2.unsqueeze(0))
        
        spectral_sim = F.cosine_similarity(
            power1[0].flatten(),
            power2[0].flatten(),
            dim=0
        ).item()
        
        return {
            "spatial_similarity": spatial_sim,
            "spectral_similarity": spectral_sim,
        }

# Made with Bob
