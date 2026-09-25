#!/usr/bin/env python3
"""
CRUMB LLM Text Generation Example

Demonstrates various text generation strategies:
- Greedy decoding
- Temperature sampling
- Top-k sampling
- Nucleus (top-p) sampling
- Streaming generation
- Batch generation

Usage:
    python examples/generate_text.py --model model.pt --prompt "BEGIN CRUMB"
    python examples/generate_text.py --model model.pt --interactive
"""

import argparse
import torch
from crumb_llm.generate import Generator, GenerationConfig
from crumb_llm.streaming import StreamingGenerator, StreamingConfig
from crumb_llm.sample import load_checkpoint

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate text with CRUMB LLM')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, default=None,
                       help='Input prompt')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode')
    parser.add_argument('--stream', action='store_true',
                       help='Stream tokens as generated')
    
    # Generation parameters
    parser.add_argument('--max-tokens', type=int, default=200,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=None,
                       help='Top-k sampling')
    parser.add_argument('--top-p', type=float, default=None,
                       help='Nucleus sampling')
    parser.add_argument('--repetition-penalty', type=float, default=1.0,
                       help='Repetition penalty')
    
    return parser.parse_args()

def generate_single(generator, prompt, config, stream=False):
    """Generate text from a single prompt."""
    if stream:
        # Streaming generation
        stream_gen = StreamingGenerator(generator.model, generator.tokenizer)
        stream_config = StreamingConfig(buffer_size=1, include_metadata=True)
        
        print(f"\n{'='*70}")
        print(f"Prompt: {prompt}")
        print(f"{'='*70}")
        print("Generated text (streaming):\n")
        
        for token in stream_gen.stream(prompt, config, stream_config):
            print(token.text, end='', flush=True)
        
        print(f"\n{'='*70}\n")
    else:
        # Standard generation
        text = generator.generate(prompt, config)
        
        print(f"\n{'='*70}")
        print(f"Prompt: {prompt}")
        print(f"{'='*70}")
        print("Generated text:\n")
        print(text)
        print(f"{'='*70}\n")

def interactive_mode(generator, config):
    """Interactive generation mode."""
    print("\n🌊 CRUMB LLM Interactive Mode")
    print("=" * 70)
    print("Enter prompts to generate text. Type 'quit' to exit.")
    print("Commands:")
    print("  /temp <value>  - Set temperature")
    print("  /tokens <n>    - Set max tokens")
    print("  /help          - Show this help")
    print("=" * 70)
    
    while True:
        try:
            prompt = input("\n> ").strip()
            
            if not prompt:
                continue
            
            if prompt.lower() == 'quit':
                print("\nGoodbye!")
                break
            
            # Handle commands
            if prompt.startswith('/'):
                parts = prompt.split()
                cmd = parts[0].lower()
                
                if cmd == '/help':
                    print("\nCommands:")
                    print("  /temp <value>  - Set temperature")
                    print("  /tokens <n>    - Set max tokens")
                    print("  /help          - Show this help")
                    continue
                
                elif cmd == '/temp' and len(parts) > 1:
                    try:
                        config.temperature = float(parts[1])
                        print(f"Temperature set to {config.temperature}")
                    except ValueError:
                        print("Invalid temperature value")
                    continue
                
                elif cmd == '/tokens' and len(parts) > 1:
                    try:
                        config.max_new_tokens = int(parts[1])
                        print(f"Max tokens set to {config.max_new_tokens}")
                    except ValueError:
                        print("Invalid token count")
                    continue
                
                else:
                    print(f"Unknown command: {cmd}")
                    continue
            
            # Generate
            text = generator.generate(prompt, config)
            print(f"\n{text}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")

def demo_sampling_strategies(generator):
    """Demonstrate different sampling strategies."""
    prompt = "BEGIN CRUMB\nv=1.3\nkind=task"
    
    print("\n🎯 Sampling Strategy Comparison")
    print("=" * 70)
    
    # Greedy decoding
    print("\n1. Greedy Decoding (temperature=0.0)")
    print("   Most likely tokens, deterministic")
    config = GenerationConfig(max_new_tokens=100, temperature=0.0)
    text = generator.generate(prompt, config)
    print(f"\n{text[:200]}...\n")
    
    # Temperature sampling
    print("\n2. Temperature Sampling (temperature=0.8)")
    print("   Balanced randomness")
    config = GenerationConfig(max_new_tokens=100, temperature=0.8)
    text = generator.generate(prompt, config)
    print(f"\n{text[:200]}...\n")
    
    # Top-k sampling
    print("\n3. Top-k Sampling (k=40)")
    print("   Sample from top 40 tokens")
    config = GenerationConfig(max_new_tokens=100, temperature=0.8, top_k=40)
    text = generator.generate(prompt, config)
    print(f"\n{text[:200]}...\n")
    
    # Nucleus sampling
    print("\n4. Nucleus Sampling (p=0.9)")
    print("   Sample from top 90% probability mass")
    config = GenerationConfig(max_new_tokens=100, temperature=0.8, top_p=0.9)
    text = generator.generate(prompt, config)
    print(f"\n{text[:200]}...\n")
    
    # With repetition penalty
    print("\n5. With Repetition Penalty (1.2)")
    print("   Penalize repeated tokens")
    config = GenerationConfig(
        max_new_tokens=100,
        temperature=0.8,
        top_k=40,
        repetition_penalty=1.2
    )
    text = generator.generate(prompt, config)
    print(f"\n{text[:200]}...\n")
    
    print("=" * 70)

def main():
    """Main generation function."""
    args = parse_args()
    
    print("🌊 CRUMB LLM Text Generation")
    print("=" * 70)
    
    # Load model
    print(f"\n📦 Loading model from {args.model}...")
    model, tokenizer = load_checkpoint(args.model)
    model.eval()
    print("✅ Model loaded")
    
    # Create generator
    generator = Generator(model, tokenizer)
    
    # Create generation config
    config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    
    # Interactive mode
    if args.interactive:
        interactive_mode(generator, config)
    
    # Demo mode (no prompt)
    elif args.prompt is None:
        demo_sampling_strategies(generator)
    
    # Single generation
    else:
        generate_single(generator, args.prompt, config, stream=args.stream)

if __name__ == '__main__':
    main()

# Made with Bob
