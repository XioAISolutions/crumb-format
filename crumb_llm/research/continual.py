"""Continual Learning for Wave Field LLM without catastrophic forgetting.

Implements lifelong learning techniques that allow the model to learn new tasks
sequentially while retaining performance on previous tasks.

Key techniques:
- Elastic Weight Consolidation (EWC) for wave parameters
- Progressive neural networks with wave layers
- Memory replay with wave field states
- Task-specific wave frequency allocation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..model import WaveFieldLM, WaveFieldConfig
from ..layers import WaveFieldBlock


@dataclass
class ContinualConfig:
    """Configuration for continual learning."""
    ewc_lambda: float = 1000.0  # EWC regularization strength
    memory_size: int = 1000  # Number of examples to store per task
    replay_batch_size: int = 32
    task_embedding_dim: int = 64
    progressive_columns: int = 3  # Number of parallel columns for progressive nets


class EWCRegularizer:
    """Elastic Weight Consolidation regularizer.
    
    Computes Fisher Information Matrix to identify important parameters
    for previous tasks and penalizes changes to them.
    """
    
    def __init__(self, model: nn.Module, lambda_: float = 1000.0):
        self.model = model
        self.lambda_ = lambda_
        
        # Store parameter importance (Fisher diagonal) and optimal values
        self.fisher: Dict[str, Tensor] = {}
        self.optimal_params: Dict[str, Tensor] = {}
    
    def compute_fisher(
        self,
        data_loader,
        num_samples: int = 1000,
    ) -> None:
        """Compute Fisher Information Matrix diagonal.
        
        Args:
            data_loader: DataLoader for the task
            num_samples: Number of samples to use
        """
        self.model.eval()
        
        # Initialize Fisher dict
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher[name] = torch.zeros_like(param)
        
        # Accumulate gradients
        samples_seen = 0
        for batch in data_loader:
            if samples_seen >= num_samples:
                break
            
            input_ids = batch["input_ids"]
            targets = batch.get("targets", input_ids)
            
            # Forward pass
            out = self.model(input_ids, targets=targets)
            loss = out["loss"]
            
            # Backward to get gradients
            self.model.zero_grad()
            loss.backward()
            
            # Accumulate squared gradients (Fisher diagonal approximation)
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher[name] += param.grad.pow(2)
            
            samples_seen += input_ids.shape[0]
        
        # Normalize by number of samples
        for name in self.fisher:
            self.fisher[name] /= samples_seen
        
        # Store optimal parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.optimal_params[name] = param.data.clone()
    
    def penalty(self) -> Tensor:
        """Compute EWC penalty for current parameters.
        
        Returns:
            penalty: Scalar penalty term
        """
        loss = 0.0
        for name, param in self.model.named_parameters():
            if name in self.fisher:
                # EWC penalty: λ/2 * F * (θ - θ*)²
                loss += (self.fisher[name] * (param - self.optimal_params[name]).pow(2)).sum()
        
        return self.lambda_ / 2 * loss


class TaskMemory:
    """Memory buffer for storing examples from previous tasks.
    
    Implements experience replay to prevent catastrophic forgetting.
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer: List[Dict[str, Tensor]] = []
        self.task_boundaries: List[int] = [0]
    
    def add_task_data(self, data: List[Dict[str, Tensor]]) -> None:
        """Add data from a new task to memory.
        
        Args:
            data: List of examples (dicts with 'input_ids', 'targets', etc.)
        """
        # Sample if too many examples
        if len(data) > self.max_size:
            indices = torch.randperm(len(data))[:self.max_size]
            data = [data[i] for i in indices]
        
        self.buffer.extend(data)
        self.task_boundaries.append(len(self.buffer))
        
        # Trim if exceeds max size
        if len(self.buffer) > self.max_size * len(self.task_boundaries):
            # Keep proportional samples from each task
            samples_per_task = self.max_size
            new_buffer = []
            for i in range(len(self.task_boundaries) - 1):
                start = self.task_boundaries[i]
                end = self.task_boundaries[i + 1]
                task_data = self.buffer[start:end]
                if len(task_data) > samples_per_task:
                    indices = torch.randperm(len(task_data))[:samples_per_task]
                    task_data = [task_data[j] for j in indices]
                new_buffer.extend(task_data)
            self.buffer = new_buffer
    
    def sample(self, batch_size: int) -> List[Dict[str, Tensor]]:
        """Sample a batch from memory.
        
        Args:
            batch_size: Number of examples to sample
            
        Returns:
            batch: List of examples
        """
        if len(self.buffer) == 0:
            return []
        
        indices = torch.randint(0, len(self.buffer), (batch_size,))
        return [self.buffer[i] for i in indices]


class ProgressiveWaveField(nn.Module):
    """Progressive neural network with wave field columns.
    
    Each new task gets a new column of wave field blocks, with lateral
    connections from previous columns.
    """
    
    def __init__(
        self,
        base_config: WaveFieldConfig,
        num_columns: int = 3,
    ):
        super().__init__()
        self.base_config = base_config
        self.num_columns = num_columns
        
        # Shared embedding
        self.embed = nn.Embedding(base_config.vocab_size, base_config.dim)
        
        # Columns of wave field blocks
        self.columns = nn.ModuleList()
        for _ in range(num_columns):
            column = nn.ModuleList([
                WaveFieldBlock(WaveFieldBlockConfig(
                    dim=base_config.dim,
                    n_heads=base_config.n_heads,
                    field_size=base_config.field_size,
                    ffn_mult=base_config.ffn_mult,
                    causal=base_config.causal,
                ))
                for _ in range(base_config.n_layers)
            ])
            self.columns.append(column)
        
        # Lateral connections between columns
        self.lateral = nn.ModuleList()
        for col_idx in range(1, num_columns):
            lateral_col = nn.ModuleList([
                nn.Linear(base_config.dim * col_idx, base_config.dim)
                for _ in range(base_config.n_layers)
            ])
            self.lateral.append(lateral_col)
        
        # Output heads per column
        self.output_heads = nn.ModuleList([
            nn.Linear(base_config.dim, base_config.vocab_size, bias=False)
            for _ in range(num_columns)
        ])
    
    def forward(
        self,
        input_ids: Tensor,
        column_idx: int = 0,
        targets: Optional[Tensor] = None,
    ) -> dict:
        """Forward through specific column with lateral connections.
        
        Args:
            input_ids: [B, N] token ids
            column_idx: Which column to use (task index)
            targets: Optional targets for loss
            
        Returns:
            dict with 'logits' and optional 'loss'
        """
        x = self.embed(input_ids)
        
        # Process through layers with lateral connections
        for layer_idx in range(self.base_config.n_layers):
            # Collect outputs from previous columns at this layer
            prev_outputs = []
            for prev_col in range(column_idx):
                with torch.no_grad():  # Don't update previous columns
                    prev_x = self.columns[prev_col][layer_idx](x)
                prev_outputs.append(prev_x)
            
            # Current column
            x = self.columns[column_idx][layer_idx](x)
            
            # Add lateral connections
            if prev_outputs:
                lateral_input = torch.cat(prev_outputs, dim=-1)
                lateral_contrib = self.lateral[column_idx - 1][layer_idx](lateral_input)
                x = x + lateral_contrib
        
        # Output head
        logits = self.output_heads[column_idx](x)
        
        out = {"logits": logits}
        if targets is not None:
            out["loss"] = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return out


class ContinualWaveField:
    """Complete continual learning system for Wave Field LLM.
    
    Combines multiple techniques:
    - EWC for parameter regularization
    - Memory replay for experience retention
    - Progressive networks for task-specific capacity
    - Task-specific wave frequency allocation
    """
    
    def __init__(
        self,
        model_config: WaveFieldConfig,
        continual_config: ContinualConfig,
        use_progressive: bool = False,
    ):
        self.model_config = model_config
        self.continual_config = continual_config
        self.use_progressive = use_progressive
        
        # Initialize model
        if use_progressive:
            self.model = ProgressiveWaveField(
                model_config,
                continual_config.progressive_columns,
            )
        else:
            self.model = WaveFieldLM(model_config)
        
        # EWC regularizer
        self.ewc = EWCRegularizer(self.model, continual_config.ewc_lambda)
        
        # Task memory
        self.memory = TaskMemory(continual_config.memory_size)
        
        # Task tracking
        self.current_task = 0
        self.task_metrics: List[Dict[str, float]] = []
    
    def train_task(
        self,
        task_data_loader,
        num_epochs: int = 10,
        learning_rate: float = 1e-4,
    ) -> Dict[str, float]:
        """Train on a new task.
        
        Args:
            task_data_loader: DataLoader for the task
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            
        Returns:
            metrics: Training metrics
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Collect task data for memory
        task_examples = []
        
        metrics = {"task_loss": 0.0, "ewc_penalty": 0.0, "replay_loss": 0.0}
        num_batches = 0
        
        for epoch in range(num_epochs):
            for batch in task_data_loader:
                input_ids = batch["input_ids"]
                targets = batch.get("targets", input_ids)
                
                # Store examples for memory
                if len(task_examples) < self.continual_config.memory_size:
                    for i in range(input_ids.shape[0]):
                        task_examples.append({
                            "input_ids": input_ids[i],
                            "targets": targets[i],
                        })
                
                # Forward pass on current task
                if self.use_progressive:
                    out = self.model(input_ids, self.current_task, targets)
                else:
                    out = self.model(input_ids, targets=targets)
                
                task_loss = out["loss"]
                
                # EWC penalty (if not first task)
                ewc_penalty = 0.0
                if self.current_task > 0 and not self.use_progressive:
                    ewc_penalty = self.ewc.penalty()
                
                # Memory replay loss
                replay_loss = 0.0
                if len(self.memory.buffer) > 0:
                    replay_batch = self.memory.sample(
                        self.continual_config.replay_batch_size
                    )
                    if replay_batch:
                        replay_ids = torch.stack([ex["input_ids"] for ex in replay_batch])
                        replay_targets = torch.stack([ex["targets"] for ex in replay_batch])
                        
                        if self.use_progressive:
                            # Replay through appropriate columns
                            replay_losses = []
                            for task_idx in range(self.current_task):
                                replay_out = self.model(replay_ids, task_idx, replay_targets)
                                replay_losses.append(replay_out["loss"])
                            replay_loss = sum(replay_losses) / len(replay_losses)
                        else:
                            replay_out = self.model(replay_ids, targets=replay_targets)
                            replay_loss = replay_out["loss"]
                
                # Total loss
                total_loss = task_loss + ewc_penalty + replay_loss
                
                # Backward and optimize
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                # Track metrics
                metrics["task_loss"] += task_loss.item()
                metrics["ewc_penalty"] += ewc_penalty if isinstance(ewc_penalty, float) else ewc_penalty.item()
                metrics["replay_loss"] += replay_loss if isinstance(replay_loss, float) else replay_loss.item()
                num_batches += 1
        
        # Average metrics
        for key in metrics:
            metrics[key] /= num_batches
        
        # Update EWC after task (if not using progressive)
        if not self.use_progressive:
            print(f"Computing Fisher information for task {self.current_task}...")
            self.ewc.compute_fisher(task_data_loader)
        
        # Add task data to memory
        self.memory.add_task_data(task_examples)
        
        # Move to next task
        self.current_task += 1
        self.task_metrics.append(metrics)
        
        return metrics
    
    def evaluate_all_tasks(
        self,
        task_data_loaders: List,
    ) -> Dict[int, float]:
        """Evaluate on all seen tasks.
        
        Args:
            task_data_loaders: List of DataLoaders, one per task
            
        Returns:
            accuracies: Dict mapping task index to accuracy
        """
        self.model.eval()
        accuracies = {}
        
        with torch.no_grad():
            for task_idx, data_loader in enumerate(task_data_loaders[:self.current_task]):
                correct = 0
                total = 0
                
                for batch in data_loader:
                    input_ids = batch["input_ids"]
                    targets = batch.get("targets", input_ids)
                    
                    if self.use_progressive:
                        out = self.model(input_ids, task_idx)
                    else:
                        out = self.model(input_ids)
                    
                    logits = out["logits"]
                    preds = logits.argmax(dim=-1)
                    
                    correct += (preds == targets).sum().item()
                    total += targets.numel()
                
                accuracies[task_idx] = correct / total if total > 0 else 0.0
        
        self.model.train()
        return accuracies
    
    def compute_forgetting(
        self,
        task_data_loaders: List,
        initial_accuracies: Dict[int, float],
    ) -> float:
        """Compute average forgetting across tasks.
        
        Args:
            task_data_loaders: List of task DataLoaders
            initial_accuracies: Initial accuracy on each task after training
            
        Returns:
            avg_forgetting: Average forgetting metric
        """
        current_accuracies = self.evaluate_all_tasks(task_data_loaders)
        
        forgetting = []
        for task_idx in range(self.current_task - 1):  # Exclude current task
            if task_idx in initial_accuracies and task_idx in current_accuracies:
                forget = initial_accuracies[task_idx] - current_accuracies[task_idx]
                forgetting.append(max(0.0, forget))  # Only positive forgetting
        
        return sum(forgetting) / len(forgetting) if forgetting else 0.0

# Made with Bob
