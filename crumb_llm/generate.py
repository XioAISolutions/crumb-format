"""Text generation with comprehensive optimization strategies.

This module provides a unified Generator class that supports multiple sampling
strategies, field-state caching, batched generation, and streaming output.
It serves as the primary interface for inference with Wave Field LLMs.

Features:
- Multiple sampling strategies (greedy, top-k, top-p, temperature, beam search)
- Field-state caching for O(F log F) per-token generation
- Batched generation with dynamic batching
- Streaming generation (yield tokens as generated)
- Flexible stop conditions (max length, EOS token, custom sequences)
- CLI interface for interactive and batch generation

Usage:
    # Interactive generation
    python -m crumb_llm.generate --model checkpoints/model.pt --interactive

    # Single generation
    python -m crumb_llm.generate --model checkpoints/model.pt --prompt "Once upon a time"

    # Programmatic usage
    from crumb_llm.generate import Generator, GenerationConfig
    
    generator = Generator(model, tokenizer)
    config = GenerationConfig(max_new_tokens=100, temperature=0.8, top_k=40)
    output = generator.generate("Hello world", config)
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Iterator, List, Union, Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from .cache import FieldStateCache, generate_cached
from .model import WaveFieldLM


@dataclass
class GenerationConfig:
    """Configuration for text generation.
    
    Attributes:
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Keep only top k tokens for sampling (None = no filtering)
        top_p: Nucleus sampling - keep tokens with cumulative prob >= p
        repetition_penalty: Penalty for repeating tokens (1.0 = no penalty)
        length_penalty: Penalty for sequence length in beam search
        num_beams: Number of beams for beam search (1 = greedy/sampling)
        early_stopping: Stop beam search when num_beams sentences are done
        do_sample: Whether to use sampling (vs greedy/beam search)
        eos_token_id: Token ID that ends generation
        pad_token_id: Token ID for padding
        stop_sequences: List of token sequences that stop generation
        min_new_tokens: Minimum tokens to generate before stopping
        use_cache: Whether to use field-state caching
        stream: Whether to yield tokens as they're generated
    """
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    num_beams: int = 1
    early_stopping: bool = False
    do_sample: bool = True
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    stop_sequences: List[List[int]] = field(default_factory=list)
    min_new_tokens: int = 0
    use_cache: bool = True
    stream: bool = False


class Generator:
    """Unified text generation interface for Wave Field LLMs.
    
    Supports multiple sampling strategies, caching, batching, and streaming.
    """
    
    def __init__(
        self,
        model: WaveFieldLM,
        tokenizer: Optional[object] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """Initialize generator.
        
        Args:
            model: Wave Field LM model
            tokenizer: Tokenizer (optional, for string I/O)
            device: Device to run on (defaults to model's device)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, Tensor, List[int]],
        config: Optional[GenerationConfig] = None,
    ) -> Union[str, Tensor, Iterator[Union[str, int]]]:
        """Generate text from a prompt.
        
        Args:
            prompt: Input prompt (string, token IDs, or tensor)
            config: Generation configuration
            
        Returns:
            Generated text (string if tokenizer provided, else token IDs)
            If config.stream=True, returns an iterator
        """
        config = config or GenerationConfig()
        
        # Convert prompt to tensor
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError("Tokenizer required for string prompts")
            input_ids = torch.tensor(
                self.tokenizer.encode(prompt),
                dtype=torch.long,
                device=self.device
            ).unsqueeze(0)
        elif isinstance(prompt, list):
            input_ids = torch.tensor(prompt, dtype=torch.long, device=self.device).unsqueeze(0)
        else:
            input_ids = prompt.to(self.device)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
        
        # Select generation strategy
        if config.stream:
            return self._generate_streaming(input_ids, config)
        elif config.num_beams > 1:
            output_ids = self._generate_beam_search(input_ids, config)
        elif config.use_cache:
            output_ids = self._generate_cached(input_ids, config)
        else:
            output_ids = self._generate_autoregressive(input_ids, config)
        
        # Convert back to string if tokenizer available
        if self.tokenizer is not None:
            return self.tokenizer.decode(output_ids[0].tolist())
        return output_ids
    
    def _generate_autoregressive(
        self,
        input_ids: Tensor,
        config: GenerationConfig,
    ) -> Tensor:
        """Standard autoregressive generation without caching.
        
        Useful for debugging or when cache overhead isn't worth it.
        """
        B, T = input_ids.shape
        generated = input_ids.clone()
        
        for step in range(config.max_new_tokens):
            # Forward pass
            logits = self.model(generated)["logits"][:, -1, :]
            
            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(
                    logits, generated, config.repetition_penalty
                )
            
            # Sample next token
            next_token = self._sample_next_token(logits, config)
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check stopping conditions
            if self._should_stop(generated, next_token, config, step):
                break
        
        return generated
    
    def _generate_cached(
        self,
        input_ids: Tensor,
        config: GenerationConfig,
    ) -> Tensor:
        """Optimized generation with field-state caching.
        
        Uses the existing generate_cached function from cache.py.
        """
        return generate_cached(
            self.model,
            input_ids,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            eos_token_id=config.eos_token_id,
        )
    
    def _generate_beam_search(
        self,
        input_ids: Tensor,
        config: GenerationConfig,
    ) -> Tensor:
        """Beam search generation.
        
        Maintains multiple hypotheses and selects the best one.
        """
        B, T = input_ids.shape
        num_beams = config.num_beams
        vocab_size = self.model.cfg.vocab_size
        
        # Expand input for beam search [B*num_beams, T]
        input_ids = input_ids.repeat_interleave(num_beams, dim=0)
        
        # Initialize beam scores
        beam_scores = torch.zeros(B, num_beams, device=self.device)
        beam_scores[:, 1:] = -1e9  # Only first beam is active initially
        
        # Track finished beams
        done = torch.zeros(B, num_beams, dtype=torch.bool, device=self.device)
        
        generated = input_ids.clone()
        
        for step in range(config.max_new_tokens):
            # Forward pass
            logits = self.model(generated)["logits"][:, -1, :]
            logits = logits.view(B, num_beams, vocab_size)
            
            # Apply temperature
            if config.temperature != 1.0:
                logits = logits / config.temperature
            
            # Compute log probabilities
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Add beam scores
            scores = beam_scores.unsqueeze(-1) + log_probs
            scores = scores.view(B, -1)  # [B, num_beams * vocab_size]
            
            # Select top num_beams candidates
            top_scores, top_indices = torch.topk(scores, num_beams, dim=-1)
            
            # Compute beam and token indices
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size
            
            # Update sequences
            new_generated = []
            for b in range(B):
                batch_beams = []
                for i in range(num_beams):
                    beam_idx = beam_indices[b, i]
                    token_idx = token_indices[b, i]
                    old_seq = generated[b * num_beams + beam_idx]
                    new_seq = torch.cat([old_seq, token_idx.unsqueeze(0)])
                    batch_beams.append(new_seq)
                new_generated.extend(batch_beams)
            
            generated = torch.stack(new_generated)
            beam_scores = top_scores
            
            # Check for EOS tokens
            if config.eos_token_id is not None:
                eos_mask = token_indices == config.eos_token_id
                done = done | eos_mask
                
                if config.early_stopping and done.all():
                    break
        
        # Select best beam for each batch
        best_beam_indices = beam_scores.argmax(dim=-1)
        output = []
        for b in range(B):
            output.append(generated[b * num_beams + best_beam_indices[b]])
        
        return torch.stack(output)
    
    def _generate_streaming(
        self,
        input_ids: Tensor,
        config: GenerationConfig,
    ) -> Iterator[Union[str, int]]:
        """Generate tokens one at a time, yielding as we go.
        
        Yields:
            Token strings (if tokenizer available) or token IDs
        """
        B, T = input_ids.shape
        if B > 1:
            raise ValueError("Streaming only supports batch size 1")
        
        # Initialize cache
        cache = FieldStateCache.init(self.model, batch_size=1, device=self.device)
        
        # Prefill phase
        x = self.model.embed(input_ids)
        for layer_idx, block in enumerate(self.model.blocks):
            x = block(x)
        
        x = self.model.norm_out(x)
        if self.model.lm_head is None:
            logits = x @ self.model.embed.weight.t()
        else:
            logits = self.model.lm_head(x)
        
        last_logits = logits[:, -1, :]
        cache.seq_pos = T
        
        generated = input_ids.clone()
        
        # Generation loop
        for step in range(config.max_new_tokens):
            # Sample next token
            next_token = self._sample_next_token(last_logits, config)
            generated = torch.cat([generated, next_token], dim=1)
            
            # Yield token
            token_id = next_token[0, 0].item()
            if self.tokenizer is not None:
                yield self.tokenizer.decode([token_id])
            else:
                yield token_id
            
            # Check stopping
            if self._should_stop(generated, next_token, config, step):
                break
            
            # Decode step with cache
            from .cache import forward_cached_block
            
            x_tok = self.model.embed(next_token)
            total_tokens = cache.seq_pos + 1
            field_size = self.model.cfg.field_size
            
            for layer_idx, block in enumerate(self.model.blocks):
                x_tok, cache.layers[layer_idx] = forward_cached_block(
                    block, x_tok, cache.layers[layer_idx],
                    token_pos=cache.seq_pos % field_size,
                    total_tokens=min(total_tokens, field_size),
                )
            
            x_tok = self.model.norm_out(x_tok)
            if self.model.lm_head is None:
                last_logits = (x_tok @ self.model.embed.weight.t()).squeeze(1)
            else:
                last_logits = self.model.lm_head(x_tok).squeeze(1)
            
            cache.seq_pos += 1
    
    def _sample_next_token(
        self,
        logits: Tensor,
        config: GenerationConfig,
    ) -> Tensor:
        """Sample next token from logits using configured strategy.
        
        Args:
            logits: [B, vocab_size] logits
            config: Generation configuration
            
        Returns:
            [B, 1] next token IDs
        """
        # Apply temperature
        if config.temperature != 1.0:
            logits = logits / max(config.temperature, 1e-6)
        
        if not config.do_sample:
            # Greedy decoding
            return logits.argmax(dim=-1, keepdim=True)
        
        # Top-k filtering
        if config.top_k is not None and config.top_k > 0:
            indices_to_remove = logits < torch.topk(logits, config.top_k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        # Top-p (nucleus) filtering
        if config.top_p is not None and config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > config.top_p
            # Keep at least one token
            sorted_indices_to_remove[..., 0] = False
            
            # Scatter back to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        # Sample from distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token
    
    def _apply_repetition_penalty(
        self,
        logits: Tensor,
        generated: Tensor,
        penalty: float,
    ) -> Tensor:
        """Apply repetition penalty to logits.
        
        Reduces probability of tokens that have already been generated.
        """
        if penalty == 1.0:
            return logits
        
        for b in range(logits.shape[0]):
            for token_id in set(generated[b].tolist()):
                # If score < 0, multiply by penalty; if > 0, divide
                if logits[b, token_id] < 0:
                    logits[b, token_id] *= penalty
                else:
                    logits[b, token_id] /= penalty
        
        return logits
    
    def _should_stop(
        self,
        generated: Tensor,
        next_token: Tensor,
        config: GenerationConfig,
        step: int,
    ) -> bool:
        """Check if generation should stop.
        
        Args:
            generated: Full generated sequence so far
            next_token: Most recently generated token
            config: Generation configuration
            step: Current generation step
            
        Returns:
            True if generation should stop
        """
        # Minimum length not reached
        if step < config.min_new_tokens:
            return False
        
        # EOS token
        if config.eos_token_id is not None:
            if (next_token == config.eos_token_id).any():
                return True
        
        # Custom stop sequences
        for stop_seq in config.stop_sequences:
            seq_len = len(stop_seq)
            if generated.shape[1] >= seq_len:
                if (generated[0, -seq_len:] == torch.tensor(stop_seq, device=generated.device)).all():
                    return True
        
        return False
    
    def generate_batch(
        self,
        prompts: List[Union[str, List[int]]],
        config: Optional[GenerationConfig] = None,
    ) -> List[Union[str, Tensor]]:
        """Generate text for multiple prompts in a batch.
        
        Args:
            prompts: List of prompts
            config: Generation configuration
            
        Returns:
            List of generated texts
        """
        config = config or GenerationConfig()
        
        # Convert all prompts to tensors
        input_ids_list = []
        for prompt in prompts:
            if isinstance(prompt, str):
                if self.tokenizer is None:
                    raise ValueError("Tokenizer required for string prompts")
                ids = self.tokenizer.encode(prompt)
            else:
                ids = prompt
            input_ids_list.append(torch.tensor(ids, dtype=torch.long, device=self.device))
        
        # Pad to same length
        max_len = max(ids.shape[0] for ids in input_ids_list)
        pad_id = config.pad_token_id or 0
        
        padded = []
        for ids in input_ids_list:
            if ids.shape[0] < max_len:
                padding = torch.full(
                    (max_len - ids.shape[0],),
                    pad_id,
                    dtype=torch.long,
                    device=self.device
                )
                ids = torch.cat([ids, padding])
            padded.append(ids)
        
        input_ids = torch.stack(padded)
        
        # Generate
        output_ids = self.generate(input_ids, config)
        
        # Convert back to strings if needed
        if self.tokenizer is not None:
            return [self.tokenizer.decode(ids.tolist()) for ids in output_ids]
        return [ids for ids in output_ids]


def load_model_and_tokenizer(checkpoint_path: Union[str, Path]):
    """Load model and tokenizer from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint directory or file
        
    Returns:
        Tuple of (model, tokenizer)
    """
    from .sample import load_checkpoint
    return load_checkpoint(checkpoint_path)


def interactive_generation(
    model: WaveFieldLM,
    tokenizer: object,
    config: Optional[GenerationConfig] = None,
):
    """Interactive generation loop.
    
    Args:
        model: Wave Field LM model
        tokenizer: Tokenizer
        config: Generation configuration
    """
    config = config or GenerationConfig()
    generator = Generator(model, tokenizer)
    
    print("Wave Field LLM Interactive Generation")
    print("=" * 50)
    print("Commands:")
    print("  /quit - Exit")
    print("  /config - Show current configuration")
    print("  /temp <value> - Set temperature")
    print("  /topk <value> - Set top-k")
    print("  /topp <value> - Set top-p")
    print("  /stream - Toggle streaming mode")
    print("=" * 50)
    print()
    
    while True:
        try:
            prompt = input(">>> ")
            
            if not prompt:
                continue
            
            # Handle commands
            if prompt.startswith("/"):
                cmd = prompt.split()[0].lower()
                
                if cmd == "/quit":
                    break
                elif cmd == "/config":
                    print(f"Temperature: {config.temperature}")
                    print(f"Top-k: {config.top_k}")
                    print(f"Top-p: {config.top_p}")
                    print(f"Max tokens: {config.max_new_tokens}")
                    print(f"Streaming: {config.stream}")
                    continue
                elif cmd == "/temp" and len(prompt.split()) > 1:
                    config.temperature = float(prompt.split()[1])
                    print(f"Temperature set to {config.temperature}")
                    continue
                elif cmd == "/topk" and len(prompt.split()) > 1:
                    val = prompt.split()[1]
                    config.top_k = None if val.lower() == "none" else int(val)
                    print(f"Top-k set to {config.top_k}")
                    continue
                elif cmd == "/topp" and len(prompt.split()) > 1:
                    val = prompt.split()[1]
                    config.top_p = None if val.lower() == "none" else float(val)
                    print(f"Top-p set to {config.top_p}")
                    continue
                elif cmd == "/stream":
                    config.stream = not config.stream
                    print(f"Streaming {'enabled' if config.stream else 'disabled'}")
                    continue
                else:
                    print(f"Unknown command: {cmd}")
                    continue
            
            # Generate
            start_time = time.time()
            
            if config.stream:
                print("", end="", flush=True)
                for token in generator.generate(prompt, config):
                    print(token, end="", flush=True)
                print()
            else:
                output = generator.generate(prompt, config)
                print(output)
            
            elapsed = time.time() - start_time
            print(f"\n[Generated in {elapsed:.2f}s]")
            print()
            
        except KeyboardInterrupt:
            print("\nInterrupted")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


def main(argv: Optional[List[str]] = None):
    """CLI entry point for generation."""
    parser = argparse.ArgumentParser(
        description="Generate text with Wave Field LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Model arguments
    parser.add_argument("--model", "--checkpoint", required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--device", default=None,
                       help="Device to run on (cuda/cpu)")
    
    # Generation mode
    parser.add_argument("--interactive", action="store_true",
                       help="Interactive generation mode")
    parser.add_argument("--prompt", default="",
                       help="Prompt for single generation")
    parser.add_argument("--prompts-file", type=Path,
                       help="File with prompts (one per line)")
    
    # Generation parameters
    parser.add_argument("--max-new-tokens", type=int, default=100,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=None,
                       help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=None,
                       help="Top-p (nucleus) sampling")
    parser.add_argument("--num-beams", type=int, default=1,
                       help="Number of beams for beam search")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                       help="Repetition penalty")
    parser.add_argument("--no-sample", action="store_true",
                       help="Use greedy decoding instead of sampling")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable field-state caching")
    parser.add_argument("--stream", action="store_true",
                       help="Stream tokens as generated")
    
    # Output
    parser.add_argument("--output", type=Path,
                       help="Output file for generated text")
    
    args = parser.parse_args(argv)
    
    # Load model
    print(f"Loading model from {args.model}...")
    model, tokenizer = load_model_and_tokenizer(args.model)
    
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    print(f"Model loaded on {device}")
    
    # Create config
    config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        do_sample=not args.no_sample,
        use_cache=not args.no_cache,
        stream=args.stream,
    )
    
    # Interactive mode
    if args.interactive:
        interactive_generation(model, tokenizer, config)
        return
    
    # Batch mode from file
    if args.prompts_file:
        prompts = args.prompts_file.read_text().strip().split("\n")
        generator = Generator(model, tokenizer, device)
        outputs = generator.generate_batch(prompts, config)
        
        if args.output:
            args.output.write_text("\n".join(outputs))
            print(f"Wrote {len(outputs)} generations to {args.output}")
        else:
            for i, output in enumerate(outputs):
                print(f"\n=== Prompt {i+1} ===")
                print(output)
        return
    
    # Single generation
    if not args.prompt:
        print("Error: Provide --prompt, --prompts-file, or --interactive")
        sys.exit(1)
    
    generator = Generator(model, tokenizer, device)
    
    if config.stream:
        print("", end="", flush=True)
        for token in generator.generate(args.prompt, config):
            print(token, end="", flush=True)
        print()
    else:
        output = generator.generate(args.prompt, config)
        print(output)
    
    if args.output:
        args.output.write_text(output if isinstance(output, str) else str(output))
        print(f"\nWrote output to {args.output}")


if __name__ == "__main__":
    main()

# Made with Bob
