"""Smoke tests for Phase 2 training infrastructure.

Tests:
- Optimizer configuration and parameter grouping
- Loss functions (cross-entropy, auxiliary losses)
- Metrics computation
- Checkpoint save/load
- Trainer initialization and basic training loop
"""

import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from crumb_llm.checkpoint import CheckpointManager, CheckpointMetadata, save_checkpoint, load_checkpoint
from crumb_llm.loss import LossConfig, WaveFieldLoss, compute_perplexity, compute_bits_per_token
from crumb_llm.metrics import MetricsAccumulator, compute_token_accuracy, compute_top_k_accuracy
from crumb_llm.model import WaveFieldLM, WaveFieldConfig
from crumb_llm.optimizer import OptimizerConfig, configure_optimizer, get_lr_scheduler


def test_optimizer_configuration():
    """Test optimizer setup with parameter grouping."""
    # Create a small model
    config = WaveFieldConfig(
        vocab_size=256,
        dim=64,
        n_layers=2,
        n_heads=2,
        field_size=128,
    )
    model = WaveFieldLM(config)
    
    # Configure optimizer
    opt_config = OptimizerConfig(
        learning_rate=1e-3,
        weight_decay=0.1,
    )
    optimizer = configure_optimizer(model, opt_config)
    
    # Check parameter groups
    assert len(optimizer.param_groups) == 2
    
    # Check weight decay settings
    decay_group = optimizer.param_groups[0]
    no_decay_group = optimizer.param_groups[1]
    
    assert decay_group["weight_decay"] == 0.1
    assert no_decay_group["weight_decay"] == 0.0
    
    # Check that we have parameters in both groups
    assert len(decay_group["params"]) > 0
    assert len(no_decay_group["params"]) > 0


def test_learning_rate_schedulers():
    """Test learning rate schedulers."""
    config = WaveFieldConfig(vocab_size=256, dim=32, n_layers=1, n_heads=2, field_size=64)
    model = WaveFieldLM(config)
    
    opt_config = OptimizerConfig(
        learning_rate=1e-3,
        warmup_steps=10,
        max_steps=100,
    )
    optimizer = configure_optimizer(model, opt_config)
    
    # Test cosine schedule
    opt_config.scheduler = "cosine"
    scheduler = get_lr_scheduler(optimizer, opt_config)
    
    # Check warmup
    initial_lr = scheduler.get_last_lr()[0]
    assert initial_lr < 1e-3  # Should start low
    
    for _ in range(10):
        scheduler.step()
    
    warmup_lr = scheduler.get_last_lr()[0]
    assert warmup_lr > initial_lr  # Should increase during warmup
    
    # Test linear schedule
    opt_config.scheduler = "linear"
    optimizer = configure_optimizer(model, opt_config)
    scheduler = get_lr_scheduler(optimizer, opt_config)
    
    for _ in range(50):
        scheduler.step()
    
    mid_lr = scheduler.get_last_lr()[0]
    assert 0 < mid_lr < 1e-3


def test_loss_computation():
    """Test loss function computation."""
    loss_config = LossConfig(
        label_smoothing=0.1,
        spectral_diversity_weight=0.01,
        field_smoothness_weight=0.01,
    )
    loss_fn = WaveFieldLoss(loss_config)
    
    # Create dummy data
    B, N, V = 2, 16, 256
    logits = torch.randn(B, N, V)
    targets = torch.randint(0, V, (B, N))
    
    # Test basic loss
    losses = loss_fn(logits, targets)
    
    assert "loss" in losses
    assert "ce_loss" in losses
    assert losses["loss"].item() > 0
    
    # Test with auxiliary losses
    field_states = torch.randn(B, 4, 32, 16)  # [B, H, F, D]
    head_spectra = torch.randn(B, 4, 17, 2)  # [B, H, F//2+1, 2]
    
    losses = loss_fn(logits, targets, field_states=field_states, head_spectra=head_spectra)
    
    assert "diversity_loss" in losses
    assert "smoothness_loss" in losses
    assert losses["loss"].item() > losses["ce_loss"].item()


def test_metrics_accumulation():
    """Test metrics accumulator."""
    acc = MetricsAccumulator()
    
    # Simulate multiple batches
    for _ in range(5):
        B, N, V = 2, 16, 256
        logits = torch.randn(B, N, V)
        targets = torch.randint(0, V, (B, N))
        loss = 2.5
        
        acc.update(loss, logits, targets)
    
    # Compute final metrics
    metrics = acc.compute()
    
    assert "loss" in metrics
    assert "perplexity" in metrics
    assert "accuracy" in metrics
    assert "bpc" in metrics
    
    assert metrics["loss"] > 0
    assert metrics["perplexity"] > 1
    assert 0 <= metrics["accuracy"] <= 1


def test_token_accuracy():
    """Test token accuracy computation."""
    B, N, V = 2, 16, 256
    
    # Perfect predictions
    logits = torch.zeros(B, N, V)
    targets = torch.randint(0, V, (B, N))
    
    for b in range(B):
        for n in range(N):
            logits[b, n, targets[b, n]] = 10.0
    
    accuracy = compute_token_accuracy(logits, targets)
    assert accuracy == 1.0
    
    # Random predictions
    logits = torch.randn(B, N, V)
    accuracy = compute_token_accuracy(logits, targets)
    assert 0 <= accuracy <= 1


def test_top_k_accuracy():
    """Test top-k accuracy computation."""
    B, N, V = 2, 16, 256
    
    logits = torch.randn(B, N, V)
    targets = torch.randint(0, V, (B, N))
    
    # Top-1 should be same as regular accuracy
    top1_acc = compute_top_k_accuracy(logits, targets, k=1)
    regular_acc = compute_token_accuracy(logits, targets)
    assert abs(top1_acc - regular_acc) < 1e-6
    
    # Top-5 should be >= top-1
    top5_acc = compute_top_k_accuracy(logits, targets, k=5)
    assert top5_acc >= top1_acc


def test_checkpoint_save_load(tmp_path):
    """Test checkpoint saving and loading."""
    # Create model and optimizer
    config = WaveFieldConfig(vocab_size=256, dim=32, n_layers=1, n_heads=2, field_size=64)
    model = WaveFieldLM(config)
    
    opt_config = OptimizerConfig()
    optimizer = configure_optimizer(model, opt_config)
    
    # Save checkpoint
    checkpoint_path = tmp_path / "test_checkpoint.pt"
    metadata = {"step": 100, "loss": 2.5}
    
    save_checkpoint(checkpoint_path, model, optimizer, metadata)
    
    assert checkpoint_path.exists()
    
    # Create new model and load
    model2 = WaveFieldLM(config)
    optimizer2 = configure_optimizer(model2, opt_config)
    
    checkpoint = load_checkpoint(checkpoint_path, model2, optimizer2)
    
    assert "metadata" in checkpoint
    assert checkpoint["metadata"]["step"] == 100
    
    # Check that weights match
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)


def test_checkpoint_manager(tmp_path):
    """Test checkpoint manager with rotation."""
    manager = CheckpointManager(
        checkpoint_dir=tmp_path,
        keep_best_n=2,
        keep_last_n=2,
    )
    
    # Create model
    config = WaveFieldConfig(vocab_size=256, dim=32, n_layers=1, n_heads=2, field_size=64)
    model = WaveFieldLM(config)
    opt_config = OptimizerConfig()
    optimizer = configure_optimizer(model, opt_config)
    
    # Save multiple checkpoints
    for step in range(5):
        metadata = CheckpointMetadata(
            step=step,
            epoch=0,
            loss=3.0 - step * 0.1,  # Decreasing loss
            learning_rate=1e-3,
        )
        
        is_best = step == 4  # Last one is best
        manager.save_checkpoint(model, optimizer, metadata, is_best=is_best)
    
    # Check that best checkpoint exists
    best_path = manager.get_best_checkpoint()
    assert best_path is not None
    assert best_path.exists()
    
    # Check that we don't have too many checkpoints
    checkpoints = list(tmp_path.glob("checkpoint_step_*.pt"))
    assert len(checkpoints) <= 2  # keep_last_n


def test_perplexity_computation():
    """Test perplexity computation."""
    loss = 2.0
    ppl = compute_perplexity(torch.tensor(loss))
    
    expected = torch.exp(torch.tensor(loss))
    assert torch.allclose(ppl, expected)
    
    # Test with very high loss (should clamp)
    high_loss = torch.tensor(100.0)
    ppl = compute_perplexity(high_loss)
    assert ppl < torch.exp(torch.tensor(100.0))  # Should be clamped


def test_bits_per_token():
    """Test bits per token computation."""
    loss = 2.0
    bpt = compute_bits_per_token(torch.tensor(loss))
    
    expected = loss / torch.log(torch.tensor(2.0))
    assert torch.allclose(bpt, expected)


def test_trainer_initialization():
    """Test trainer initialization (without actual training)."""
    pytest.importorskip("tqdm")
    
    from crumb_llm.trainer import TrainerConfig, WaveFieldTrainer
    from torch.utils.data import TensorDataset, DataLoader
    
    # Create model
    config = WaveFieldConfig(vocab_size=256, dim=32, n_layers=1, n_heads=2, field_size=64)
    model = WaveFieldLM(config)
    
    # Create dummy dataset
    num_samples = 100
    seq_len = 32
    data = torch.randint(0, 256, (num_samples, seq_len))
    dataset = TensorDataset(data, data)
    train_loader = DataLoader(dataset, batch_size=4)
    eval_loader = DataLoader(dataset, batch_size=4)
    
    # Create trainer config
    trainer_config = TrainerConfig(
        max_steps=10,
        eval_every=5,
        log_every=2,
        save_every=100,  # Don't save during test
        mixed_precision=False,  # Disable for CPU testing
        device="cpu",
    )
    
    # Initialize trainer (should not raise)
    trainer = WaveFieldTrainer(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        config=trainer_config,
    )
    
    assert trainer.global_step == 0
    assert trainer.epoch == 0
    assert trainer.device.type == "cpu"


def test_loss_with_masking():
    """Test loss computation with padding mask."""
    loss_config = LossConfig(ignore_index=-100)
    loss_fn = WaveFieldLoss(loss_config)
    
    B, N, V = 2, 16, 256
    logits = torch.randn(B, N, V)
    targets = torch.randint(0, V, (B, N))
    
    # Add padding (set some targets to ignore_index)
    targets[:, -4:] = -100
    
    losses = loss_fn(logits, targets)
    
    # Loss should be computed only on non-padding tokens
    assert losses["loss"].item() > 0
    assert torch.isfinite(losses["loss"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Made with Bob
