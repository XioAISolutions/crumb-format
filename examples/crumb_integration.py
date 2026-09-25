#!/usr/bin/env python3
"""
CRUMB-Specific Integration Example

Demonstrates CRUMB LLM's native structure-aware features:
- Structure-aware training
- Priority-based generation
- Section boundary handling
- Cross-reference resolution
- Fold-aware processing

Usage:
    python examples/crumb_integration.py --mode train
    python examples/crumb_integration.py --mode generate
"""

import argparse
import glob
import torch
from crumb_llm import WaveFieldLM, WaveFieldConfig
from crumb_llm.data_enhanced import CRUMBDataset, create_dataloader, DataConfig
from crumb_llm.tokenizer import ByteTokenizer
from crumb_llm.trainer import WaveFieldTrainer, TrainerConfig
from crumb_llm.crumb_adapter import CRUMBAdapter
from crumb_llm.cache import FieldStateCache
from crumb_llm.generate import Generator, GenerationConfig

def train_structure_aware(crumb_files):
    """Train model with CRUMB structure awareness."""
    print("🌊 CRUMB-Aware Training")
    print("=" * 70)
    
    # Create tokenizer
    tokenizer = ByteTokenizer()
    
    # Create CRUMB dataset
    print(f"\n📚 Loading {len(crumb_files)} CRUMB files...")
    dataset = CRUMBDataset(
        crumb_files,
        tokenizer,
        block_size=512,
        preserve_structure=True,  # Keep section boundaries
    )
    
    print(f"   Total samples: {len(dataset)}")
    
    # Create dataloader
    data_config = DataConfig(
        batch_size=16,
        dynamic_batching=True,
        num_workers=4,
    )
    train_loader = create_dataloader(dataset, data_config, is_train=True)
    
    # Create model
    print("\n🏗️  Creating model...")
    config = WaveFieldConfig(
        vocab_size=256,
        dim=512,
        n_layers=8,
        n_heads=8,
        field_size=1024,
    )
    model = WaveFieldLM(config)
    
    # Configure structure-aware training
    print("\n⚙️  Configuring structure-aware training...")
    trainer_config = TrainerConfig(
        max_steps=10000,
        learning_rate=3e-4,
        structure_aware=True,      # Enable structure detection
        priority_weighting=True,   # Use @priority annotations
        fold_adaptive=True,        # Adaptive damping for folds
        eval_every=500,
        save_every=1000,
        checkpoint_dir='./crumb_checkpoints',
    )
    
    print("   Structure awareness: ENABLED")
    print("   Priority weighting: ENABLED")
    print("   Fold adaptation: ENABLED")
    
    # Train
    print("\n🎯 Training...")
    trainer = WaveFieldTrainer(model, train_loader, None, trainer_config)
    metrics = trainer.train()
    
    print(f"\n✅ Training complete!")
    print(f"   Final loss: {metrics['final_loss']:.4f}")
    print(f"   Structure metrics:")
    if 'section_accuracy' in metrics:
        print(f"     Section boundary accuracy: {metrics['section_accuracy']:.1%}")
    if 'priority_correlation' in metrics:
        print(f"     Priority correlation: {metrics['priority_correlation']:.2f}")
    
    return model, tokenizer

def generate_with_structure(model, tokenizer):
    """Generate CRUMB documents with structure awareness."""
    print("\n🌊 Structure-Aware Generation")
    print("=" * 70)
    
    # Wrap model with CRUMB adapter
    adapter = CRUMBAdapter(model)
    generator = Generator(adapter, tokenizer)
    
    # Example 1: Generate with priorities
    print("\n1. Priority-Aware Generation")
    print("-" * 70)
    
    prompt = """BEGIN CRUMB
v=1.3
kind=task
title=Important Task
---
[goal]
@priority: 5
"""
    
    config = GenerationConfig(
        max_new_tokens=150,
        temperature=0.8,
        use_priorities=True,  # Enable priority-aware generation
    )
    
    text = generator.generate(prompt, config)
    print(text)
    
    # Example 2: Generate with section boundaries
    print("\n2. Section-Aware Generation")
    print("-" * 70)
    
    prompt = """BEGIN CRUMB
v=1.3
kind=mem
title=Memory
---
[consolidated]
"""
    
    config = GenerationConfig(
        max_new_tokens=150,
        temperature=0.8,
        respect_boundaries=True,  # Respect section boundaries
    )
    
    text = generator.generate(prompt, config)
    print(text)
    
    # Example 3: Generate with fold awareness
    print("\n3. Fold-Aware Generation")
    print("-" * 70)
    
    prompt = """BEGIN CRUMB
v=1.3
kind=task
---
[fold:context/summary]
"""
    
    config = GenerationConfig(
        max_new_tokens=100,
        temperature=0.7,
        fold_aware=True,  # Adaptive generation for folds
    )
    
    text = generator.generate(prompt, config)
    print(text)

def demonstrate_cross_references(model, tokenizer):
    """Demonstrate cross-reference handling."""
    print("\n🌊 Cross-Reference Resolution")
    print("=" * 70)
    
    adapter = CRUMBAdapter(model)
    cache = FieldStateCache.init(model, batch_size=1)
    
    # Create reference document
    ref_doc = """BEGIN CRUMB
v=1.3
kind=mem
id=prefs-abc123
title=User Preferences
---
[consolidated]
- Prefers concise technical answers
- Likes code examples
- Uses Python and TypeScript
END CRUMB
"""
    
    print("\n1. Processing reference document...")
    ref_ids = torch.tensor([tokenizer.encode(ref_doc)])
    ref_field = adapter.process_reference(ref_ids)
    cache.store('prefs-abc123', ref_field)
    print("   ✓ Reference cached")
    
    # Generate with reference
    print("\n2. Generating with cross-reference...")
    prompt = """BEGIN CRUMB
v=1.3
kind=task
refs=prefs-abc123
title=Write Documentation
---
[goal]
Write API documentation for the new feature.

[context]
"""
    
    input_ids = torch.tensor([tokenizer.encode(prompt)])
    output = adapter.generate_with_refs(
        input_ids,
        ref_cache=cache,
        max_new_tokens=150,
        temperature=0.8,
    )
    
    text = tokenizer.decode(output[0].tolist())
    print(text)
    print("\n   ✓ Generated with reference context")

def analyze_structure_benefits(model, tokenizer):
    """Analyze benefits of structure awareness."""
    print("\n🌊 Structure Awareness Benefits")
    print("=" * 70)
    
    # Test document
    test_doc = """BEGIN CRUMB
v=1.3
kind=task
---
[goal]
@priority: 5
This is very important.

[context]
@priority: 2
This is less important background.

[constraints]
@priority: 4
These are important constraints.
END CRUMB
"""
    
    print("\n1. Computing perplexity with structure awareness...")
    adapter = CRUMBAdapter(model)
    
    input_ids = torch.tensor([tokenizer.encode(test_doc)])
    
    with torch.no_grad():
        # With structure
        outputs_struct = adapter(input_ids, use_structure=True)
        loss_struct = outputs_struct['loss'].item()
        ppl_struct = torch.exp(outputs_struct['loss']).item()
        
        # Without structure
        outputs_plain = model(input_ids)
        loss_plain = outputs_plain['loss'].item()
        ppl_plain = torch.exp(outputs_plain['loss']).item()
    
    print(f"\n   With structure awareness:")
    print(f"     Loss: {loss_struct:.4f}")
    print(f"     Perplexity: {ppl_struct:.2f}")
    
    print(f"\n   Without structure awareness:")
    print(f"     Loss: {loss_plain:.4f}")
    print(f"     Perplexity: {ppl_plain:.2f}")
    
    improvement = (ppl_plain - ppl_struct) / ppl_plain * 100
    print(f"\n   ✓ Improvement: {improvement:.1f}%")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='CRUMB Integration Examples')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'generate', 'refs', 'analyze'],
                       help='Example mode')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model (for generate/refs/analyze)')
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Train structure-aware model
        crumb_files = glob.glob('examples/*.crumb')
        if not crumb_files:
            print("Error: No CRUMB files found in examples/")
            print("Please add some .crumb files to the examples/ directory")
            return
        
        model, tokenizer = train_structure_aware(crumb_files)
        
        # Save model
        torch.save({
            'model': model.state_dict(),
            'config': model.cfg,
            'tokenizer': tokenizer,
        }, 'crumb_aware_model.pt')
        print(f"\n💾 Model saved to: crumb_aware_model.pt")
    
    elif args.mode in ['generate', 'refs', 'analyze']:
        if not args.model:
            print("Error: --model required for this mode")
            return
        
        # Load model
        print(f"📦 Loading model from {args.model}...")
        checkpoint = torch.load(args.model)
        config = checkpoint['config']
        model = WaveFieldLM(config)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        tokenizer = checkpoint['tokenizer']
        print("✅ Model loaded")
        
        if args.mode == 'generate':
            generate_with_structure(model, tokenizer)
        elif args.mode == 'refs':
            demonstrate_cross_references(model, tokenizer)
        elif args.mode == 'analyze':
            analyze_structure_benefits(model, tokenizer)
    
    print("\n🎉 Done!")

if __name__ == '__main__':
    main()

# Made with Bob
