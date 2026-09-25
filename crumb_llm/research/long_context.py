"""Long Context Extensions for Wave Field LLM.

Handle extremely long sequences (100K+ tokens) through:
- Hierarchical wave fields (local + global)
- Sliding window with wave state persistence
- Landmark attention via wave anchors
- Compressive memory with wave summarization
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..model import WaveFieldLM, WaveFieldConfig
from ..layers import WaveFieldBlock, WaveFieldBlockConfig, RMSNorm
from ..scatter_gather import scatter_linear, gather_linear


@dataclass
class LongContextConfig:
    """Configuration for long context handling."""
    window_size: int = 2048  # Local window size
    num_landmarks: int = 64  # Number of landmark tokens
    compression_ratio: int = 4  # Compression ratio for memory
    num_global_layers: int = 2  # Number of global attention layers
    recurrence_depth: int = 3  # Depth of recurrent processing


class HierarchicalField(nn.Module):
    """Hierarchical wave field with local and global scales.
    
    Processes sequences at multiple scales:
    - Local: High-resolution wave field for nearby tokens
    - Global: Compressed wave field for distant context
    """
    
    def __init__(
        self,
        config: WaveFieldConfig,
        long_config: LongContextConfig,
    ):
        super().__init__()
        self.config = config
        self.long_config = long_config
        
        # Local wave field blocks (high resolution)
        local_cfg = WaveFieldBlockConfig(
            dim=config.dim,
            n_heads=config.n_heads,
            field_size=long_config.window_size,
            ffn_mult=config.ffn_mult,
            causal=config.causal,
        )
        self.local_blocks = nn.ModuleList([
            WaveFieldBlock(local_cfg)
            for _ in range(config.n_layers - long_config.num_global_layers)
        ])
        
        # Global wave field blocks (compressed)
        global_cfg = WaveFieldBlockConfig(
            dim=config.dim,
            n_heads=config.n_heads,
            field_size=config.field_size,
            ffn_mult=config.ffn_mult,
            causal=config.causal,
        )
        self.global_blocks = nn.ModuleList([
            WaveFieldBlock(global_cfg)
            for _ in range(long_config.num_global_layers)
        ])
        
        # Compression layer (local → global)
        self.compress = nn.Linear(config.dim, config.dim)
        
        # Expansion layer (global → local)
        self.expand = nn.Linear(config.dim, config.dim)
    
    def compress_context(self, x: Tensor) -> Tensor:
        """Compress local context to global representation.
        
        Args:
            x: [B, N, D] local context
            
        Returns:
            compressed: [B, N//ratio, D] compressed context
        """
        B, N, D = x.shape
        ratio = self.long_config.compression_ratio
        
        # Reshape and pool
        if N % ratio != 0:
            # Pad to multiple of ratio
            pad_len = ratio - (N % ratio)
            x = F.pad(x, (0, 0, 0, pad_len))
            N = x.shape[1]
        
        x_reshaped = x.view(B, N // ratio, ratio, D)
        compressed = x_reshaped.mean(dim=2)  # Average pool
        
        return self.compress(compressed)
    
    def expand_context(self, x: Tensor, target_length: int) -> Tensor:
        """Expand global context back to local resolution.
        
        Args:
            x: [B, N_compressed, D] compressed context
            target_length: Target sequence length
            
        Returns:
            expanded: [B, target_length, D] expanded context
        """
        B, N_comp, D = x.shape
        
        # Expand via interpolation
        x_expanded = self.expand(x)
        
        # Upsample to target length
        x_expanded = x_expanded.transpose(1, 2)  # [B, D, N_comp]
        x_expanded = F.interpolate(
            x_expanded,
            size=target_length,
            mode='linear',
            align_corners=False,
        )
        x_expanded = x_expanded.transpose(1, 2)  # [B, target_length, D]
        
        return x_expanded
    
    def forward(self, x: Tensor) -> Tensor:
        """Hierarchical forward pass.
        
        Args:
            x: [B, N, D] input (can be very long)
            
        Returns:
            output: [B, N, D] processed output
        """
        B, N, D = x.shape
        
        # Process with local blocks (sliding window)
        window_size = self.long_config.window_size
        
        if N <= window_size:
            # Short sequence: process normally
            for block in self.local_blocks:
                x = block(x)
        else:
            # Long sequence: sliding window
            outputs = []
            for start in range(0, N, window_size // 2):  # 50% overlap
                end = min(start + window_size, N)
                window = x[:, start:end, :]
                
                for block in self.local_blocks:
                    window = block(window)
                
                outputs.append(window)
            
            # Merge overlapping windows (average)
            x_merged = torch.zeros_like(x)
            counts = torch.zeros(B, N, 1, device=x.device)
            
            for i, start in enumerate(range(0, N, window_size // 2)):
                end = min(start + window_size, N)
                x_merged[:, start:end, :] += outputs[i]
                counts[:, start:end, :] += 1
            
            x = x_merged / counts.clamp(min=1)
        
        # Compress to global scale
        x_global = self.compress_context(x)
        
        # Process with global blocks
        for block in self.global_blocks:
            x_global = block(x_global)
        
        # Expand back to local scale
        x_expanded = self.expand_context(x_global, N)
        
        # Residual connection
        x = x + x_expanded
        
        return x


class LandmarkAttention(nn.Module):
    """Landmark-based attention using wave anchors.
    
    Selects landmark tokens that serve as anchors for long-range
    wave propagation.
    """
    
    def __init__(self, dim: int, num_landmarks: int):
        super().__init__()
        self.dim = dim
        self.num_landmarks = num_landmarks
        
        # Landmark selection (learnable)
        self.landmark_query = nn.Parameter(torch.randn(num_landmarks, dim))
        
        # Attention layers
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
    
    def select_landmarks(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Select landmark tokens from sequence.
        
        Args:
            x: [B, N, D] input sequence
            
        Returns:
            landmarks: [B, num_landmarks, D] landmark tokens
            indices: [B, num_landmarks] landmark indices
        """
        B, N, D = x.shape
        
        # Compute similarity to landmark queries
        similarity = torch.matmul(
            self.landmark_query.unsqueeze(0),  # [1, L, D]
            x.transpose(1, 2),  # [B, D, N]
        )  # [B, L, N]
        
        # Select top tokens for each landmark
        _, indices = similarity.max(dim=-1)  # [B, L]
        
        # Gather landmarks
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, D)
        landmarks = torch.gather(x, 1, indices_expanded)
        
        return landmarks, indices
    
    def forward(self, x: Tensor) -> Tensor:
        """Landmark attention forward.
        
        Args:
            x: [B, N, D] input sequence
            
        Returns:
            output: [B, N, D] attended output
        """
        B, N, D = x.shape
        
        # Select landmarks
        landmarks, _ = self.select_landmarks(x)  # [B, L, D]
        
        # Compute attention: all tokens attend to landmarks
        q = self.q_proj(x)  # [B, N, D]
        k = self.k_proj(landmarks)  # [B, L, D]
        v = self.v_proj(landmarks)  # [B, L, D]
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(1, 2)) / (D ** 0.5)  # [B, N, L]
        attn = F.softmax(scores, dim=-1)
        
        # Attend to landmarks
        out = torch.matmul(attn, v)  # [B, N, D]
        out = self.out_proj(out)
        
        return out


class CompressiveMemory(nn.Module):
    """Compressive memory for infinite context.
    
    Maintains a compressed memory of past context that can be
    retrieved and integrated with current processing.
    """
    
    def __init__(
        self,
        dim: int,
        memory_size: int = 1024,
        compression_ratio: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.memory_size = memory_size
        self.compression_ratio = compression_ratio
        
        # Memory buffer (ring buffer)
        self.register_buffer(
            "memory",
            torch.zeros(1, memory_size, dim),
        )
        self.register_buffer("memory_ptr", torch.zeros(1, dtype=torch.long))
        
        # Compression network
        self.compressor = nn.Sequential(
            nn.Linear(dim * compression_ratio, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        
        # Memory attention
        self.memory_attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
    
    def compress_and_store(self, x: Tensor) -> None:
        """Compress and store new context in memory.
        
        Args:
            x: [B, N, D] new context to store
        """
        B, N, D = x.shape
        ratio = self.compression_ratio
        
        # Compress
        if N % ratio != 0:
            pad_len = ratio - (N % ratio)
            x = F.pad(x, (0, 0, 0, pad_len))
            N = x.shape[1]
        
        x_reshaped = x.view(B, N // ratio, ratio * D)
        compressed = self.compressor(x_reshaped)  # [B, N//ratio, D]
        
        # Store in ring buffer
        num_new = compressed.shape[1]
        ptr = self.memory_ptr.item()
        
        if ptr + num_new <= self.memory_size:
            self.memory[:, ptr:ptr + num_new, :] = compressed
            self.memory_ptr[0] = (ptr + num_new) % self.memory_size
        else:
            # Wrap around
            first_part = self.memory_size - ptr
            self.memory[:, ptr:, :] = compressed[:, :first_part, :]
            self.memory[:, :num_new - first_part, :] = compressed[:, first_part:, :]
            self.memory_ptr[0] = num_new - first_part
    
    def retrieve(self, query: Tensor, top_k: int = 64) -> Tensor:
        """Retrieve relevant memory based on query.
        
        Args:
            query: [B, N, D] query tokens
            top_k: Number of memory slots to retrieve
            
        Returns:
            retrieved: [B, N, D] retrieved and integrated memory
        """
        B, N, D = query.shape
        
        # Attend to memory
        retrieved, _ = self.memory_attn(
            query,
            self.memory.expand(B, -1, -1),
            self.memory.expand(B, -1, -1),
        )
        
        return retrieved


class LongContextWaveField(nn.Module):
    """Complete long context system for Wave Field LLM.
    
    Combines hierarchical fields, landmark attention, and compressive
    memory to handle 100K+ token sequences efficiently.
    """
    
    def __init__(
        self,
        config: WaveFieldConfig,
        long_config: LongContextConfig,
    ):
        super().__init__()
        self.config = config
        self.long_config = long_config
        
        # Embedding
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        
        # Hierarchical wave field
        self.hierarchical = HierarchicalField(config, long_config)
        
        # Landmark attention
        self.landmark_attn = LandmarkAttention(config.dim, long_config.num_landmarks)
        
        # Compressive memory
        self.memory = CompressiveMemory(
            config.dim,
            memory_size=long_config.num_landmarks * 16,
            compression_ratio=long_config.compression_ratio,
        )
        
        # Output
        self.norm_out = RMSNorm(config.dim)
        if config.tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
    
    def forward(
        self,
        input_ids: Tensor,
        targets: Optional[Tensor] = None,
        use_memory: bool = True,
    ) -> dict:
        """Forward pass with long context handling.
        
        Args:
            input_ids: [B, N] token ids (N can be very large)
            targets: Optional targets for loss
            use_memory: Whether to use compressive memory
            
        Returns:
            dict with 'logits' and optional 'loss'
        """
        B, N = input_ids.shape
        
        # Embed
        x = self.embed(input_ids)
        
        # Retrieve from memory if enabled
        if use_memory:
            memory_context = self.memory.retrieve(x)
            x = x + memory_context
        
        # Hierarchical processing
        x = self.hierarchical(x)
        
        # Landmark attention for global context
        landmark_context = self.landmark_attn(x)
        x = x + landmark_context
        
        # Store in memory for future use
        if use_memory:
            self.memory.compress_and_store(x.detach())
        
        # Output
        x = self.norm_out(x)
        
        if self.lm_head is None:
            logits = x @ self.embed.weight.t()
        else:
            logits = self.lm_head(x)
        
        out = {"logits": logits}
        
        if targets is not None:
            out["loss"] = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        
        return out
    
    @torch.no_grad()
    def generate_long(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 1000,
        temperature: float = 1.0,
        use_memory: bool = True,
    ) -> Tensor:
        """Generate with long context support.
        
        Args:
            input_ids: [B, N] prompt tokens
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            use_memory: Whether to use compressive memory
            
        Returns:
            generated: [B, N + max_new_tokens] generated sequence
        """
        ids = input_ids
        
        for _ in range(max_new_tokens):
            # Use sliding window for very long sequences
            if ids.shape[1] > self.long_config.window_size:
                context = ids[:, -self.long_config.window_size:]
            else:
                context = ids
            
            # Forward
            out = self(context, use_memory=use_memory)
            logits = out["logits"][:, -1, :] / max(temperature, 1e-6)
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            
            # Append
            ids = torch.cat([ids, next_id], dim=1)
        
        return ids

# Made with Bob
