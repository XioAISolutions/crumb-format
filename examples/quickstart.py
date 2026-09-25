#!/usr/bin/env python3
"""
CRUMB LLM Quickstart Example

Train a tiny model and generate text in ~50 lines of code.
This is the fastest way to see CRUMB LLM in action.

Usage:
    python examples/quickstart.py
"""

import torch
from crumb_llm import WaveFieldLM, WaveFieldConfig
from crumb_llm.tokenizer import ByteTokenizer
from crumb_llm.data import TextDataset

def main():
    print("🌊 CRUMB LLM Quickstart")
    print("=" * 50)
    
    # 1. Prepare data
    print("\n📚 Loading training data...")
    training_text = """BEGIN CRUMB
v=1.3
kind=task
title=Example Task
---
[goal]
This is an example CRUMB document for training.

[context]
CRUMB LLM learns to generate structured documents.
It understands section boundaries and priorities.

[constraints]
- Must follow CRUMB format
- Should be well-structured
END CRUMB
""" * 100  # Repeat for more training data
    
    tokenizer = ByteTokenizer()
    dataset = TextDataset(training_text, tokenizer, block_size=128)
    
    # 2. Create tiny model
    print("🏗️  Creating model (230K parameters)...")
    config = WaveFieldConfig.tiny()
    model = WaveFieldLM(config)
    print(f"   Model size: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 3. Train
    print("\n🎯 Training for 500 steps...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    model.train()
    for step in range(500):
        # Get batch
        idx = torch.randint(0, len(dataset), (8,))
        batch = torch.stack([dataset[i] for i in idx])
        
        # Forward pass
        outputs = model(batch[:, :-1], targets=batch[:, 1:])
        loss = outputs['loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Log progress
        if (step + 1) % 100 == 0:
            print(f"   Step {step+1}/500 | Loss: {loss.item():.4f}")
    
    print("✅ Training complete!")
    
    # 4. Generate text
    print("\n📝 Generating text...")
    model.eval()
    
    prompt = "BEGIN CRUMB\nv=1.3\nkind=task"
    context = tokenizer.encode(prompt)
    context = torch.tensor([context], dtype=torch.long)
    
    with torch.no_grad():
        generated = model.generate(
            context,
            max_new_tokens=100,
            temperature=0.8,
        )
    
    text = tokenizer.decode(generated[0].tolist())
    print("\n" + "=" * 50)
    print("Generated text:")
    print("=" * 50)
    print(text)
    print("=" * 50)
    
    # 5. Save model
    print("\n💾 Saving model...")
    torch.save({
        'model': model.state_dict(),
        'config': config,
        'tokenizer': tokenizer,
    }, 'quickstart_model.pt')
    print("   Saved to: quickstart_model.pt")
    
    print("\n🎉 Quickstart complete!")
    print("\nNext steps:")
    print("  - Train longer: Increase steps to 5000+")
    print("  - Use more data: Add your own training text")
    print("  - Try larger model: Use WaveFieldConfig.small()")
    print("  - See docs/GETTING_STARTED.md for more")

if __name__ == '__main__':
    main()

# Made with Bob
