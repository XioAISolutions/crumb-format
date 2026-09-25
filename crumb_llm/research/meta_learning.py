"""Meta-Learning and Few-Shot Learning for Wave Field LLM.

Implements rapid adaptation techniques:
- MAML (Model-Agnostic Meta-Learning)
- Prototypical networks in wave space
- Matching networks with wave similarity
- Reptile optimization
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..model import WaveFieldLM, WaveFieldConfig


@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning."""
    inner_lr: float = 0.01  # Learning rate for inner loop
    outer_lr: float = 0.001  # Learning rate for outer loop
    num_inner_steps: int = 5  # Gradient steps in inner loop
    num_shots: int = 5  # Number of examples per class (K-shot)
    num_ways: int = 5  # Number of classes (N-way)
    meta_batch_size: int = 4  # Number of tasks per meta-batch


class MAMLOptimizer:
    """Model-Agnostic Meta-Learning optimizer for wave fields.
    
    Learns initialization that can quickly adapt to new tasks
    with few gradient steps.
    """
    
    def __init__(
        self,
        model: WaveFieldLM,
        config: MetaLearningConfig,
    ):
        self.model = model
        self.config = config
        
        # Meta-optimizer (updates the initialization)
        self.meta_optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.outer_lr,
        )
    
    def inner_loop(
        self,
        support_ids: Tensor,
        support_targets: Tensor,
    ) -> Tuple[nn.Module, float]:
        """Perform inner loop adaptation on support set.
        
        Args:
            support_ids: [K, N] support examples
            support_targets: [K, N] support targets
            
        Returns:
            adapted_model: Model after adaptation
            loss: Final inner loop loss
        """
        # Clone model for adaptation
        adapted_model = type(self.model)(self.model.cfg)
        adapted_model.load_state_dict(self.model.state_dict())
        
        # Inner loop optimizer
        inner_optimizer = torch.optim.SGD(
            adapted_model.parameters(),
            lr=self.config.inner_lr,
        )
        
        # Adapt on support set
        for _ in range(self.config.num_inner_steps):
            out = adapted_model(support_ids, targets=support_targets)
            loss = out["loss"]
            
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()
        
        return adapted_model, loss.item()
    
    def meta_train_step(
        self,
        task_batch: List[Dict[str, Tensor]],
    ) -> Dict[str, float]:
        """Single meta-training step on a batch of tasks.
        
        Args:
            task_batch: List of tasks, each with 'support' and 'query' sets
            
        Returns:
            metrics: Training metrics
        """
        meta_loss = 0.0
        
        for task in task_batch:
            support_ids = task["support_ids"]
            support_targets = task["support_targets"]
            query_ids = task["query_ids"]
            query_targets = task["query_targets"]
            
            # Inner loop: adapt to support set
            adapted_model, _ = self.inner_loop(support_ids, support_targets)
            
            # Outer loop: evaluate on query set
            out = adapted_model(query_ids, targets=query_targets)
            meta_loss += out["loss"]
        
        # Average over tasks
        meta_loss = meta_loss / len(task_batch)
        
        # Meta-update
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return {"meta_loss": meta_loss.item()}
    
    def adapt_to_task(
        self,
        support_ids: Tensor,
        support_targets: Tensor,
    ) -> WaveFieldLM:
        """Adapt model to a new task.
        
        Args:
            support_ids: [K, N] support examples
            support_targets: [K, N] support targets
            
        Returns:
            adapted_model: Adapted model
        """
        adapted_model, _ = self.inner_loop(support_ids, support_targets)
        return adapted_model


class PrototypicalNetwork(nn.Module):
    """Prototypical networks in wave field space.
    
    Classifies by computing distances to class prototypes
    in the wave representation space.
    """
    
    def __init__(self, encoder: WaveFieldLM):
        super().__init__()
        self.encoder = encoder
    
    def compute_prototypes(
        self,
        support_ids: Tensor,
        support_labels: Tensor,
        num_classes: int,
    ) -> Tensor:
        """Compute class prototypes from support set.
        
        Args:
            support_ids: [K*N_way, N] support examples
            support_labels: [K*N_way] class labels
            num_classes: Number of classes
            
        Returns:
            prototypes: [N_way, D] class prototypes
        """
        # Encode support set
        x = self.encoder.embed(support_ids)
        for block in self.encoder.blocks:
            x = block(x)
        x = self.encoder.norm_out(x)
        
        # Pool over sequence
        embeddings = x.mean(dim=1)  # [K*N_way, D]
        
        # Compute prototypes (mean of each class)
        prototypes = []
        for c in range(num_classes):
            class_mask = support_labels == c
            class_embeddings = embeddings[class_mask]
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)
        
        return torch.stack(prototypes)  # [N_way, D]
    
    def forward(
        self,
        query_ids: Tensor,
        prototypes: Tensor,
    ) -> Tensor:
        """Classify queries by distance to prototypes.
        
        Args:
            query_ids: [Q, N] query examples
            prototypes: [N_way, D] class prototypes
            
        Returns:
            logits: [Q, N_way] classification logits
        """
        # Encode queries
        x = self.encoder.embed(query_ids)
        for block in self.encoder.blocks:
            x = block(x)
        x = self.encoder.norm_out(x)
        
        # Pool over sequence
        query_embeddings = x.mean(dim=1)  # [Q, D]
        
        # Compute distances to prototypes (negative Euclidean distance)
        distances = torch.cdist(query_embeddings, prototypes)  # [Q, N_way]
        logits = -distances  # Negative distance as logits
        
        return logits


class MatchingNetwork(nn.Module):
    """Matching networks with wave field similarity.
    
    Uses attention over support set to classify queries.
    """
    
    def __init__(self, encoder: WaveFieldLM):
        super().__init__()
        self.encoder = encoder
        
        # Attention for matching
        dim = encoder.cfg.dim
        self.query_proj = nn.Linear(dim, dim)
        self.support_proj = nn.Linear(dim, dim)
    
    def forward(
        self,
        query_ids: Tensor,
        support_ids: Tensor,
        support_labels: Tensor,
        num_classes: int,
    ) -> Tensor:
        """Classify queries by matching to support set.
        
        Args:
            query_ids: [Q, N] query examples
            support_ids: [K*N_way, N] support examples
            support_labels: [K*N_way] support labels
            num_classes: Number of classes
            
        Returns:
            logits: [Q, N_way] classification logits
        """
        # Encode queries
        x_q = self.encoder.embed(query_ids)
        for block in self.encoder.blocks:
            x_q = block(x_q)
        x_q = self.encoder.norm_out(x_q)
        query_emb = x_q.mean(dim=1)  # [Q, D]
        
        # Encode support
        x_s = self.encoder.embed(support_ids)
        for block in self.encoder.blocks:
            x_s = block(x_s)
        x_s = self.encoder.norm_out(x_s)
        support_emb = x_s.mean(dim=1)  # [K*N_way, D]
        
        # Project for attention
        query_proj = self.query_proj(query_emb)  # [Q, D]
        support_proj = self.support_proj(support_emb)  # [K*N_way, D]
        
        # Compute attention weights (similarity)
        attention = torch.matmul(query_proj, support_proj.t())  # [Q, K*N_way]
        attention = F.softmax(attention, dim=-1)
        
        # Weighted sum of support labels
        logits = torch.zeros(query_emb.shape[0], num_classes, device=query_ids.device)
        for c in range(num_classes):
            class_mask = (support_labels == c).float()  # [K*N_way]
            logits[:, c] = (attention * class_mask.unsqueeze(0)).sum(dim=-1)
        
        return logits


class ReptileOptimizer:
    """Reptile meta-learning optimizer.
    
    Simpler alternative to MAML that directly interpolates
    between initial and adapted parameters.
    """
    
    def __init__(
        self,
        model: WaveFieldLM,
        config: MetaLearningConfig,
    ):
        self.model = model
        self.config = config
        self.meta_lr = config.outer_lr
    
    def meta_train_step(
        self,
        task_batch: List[Dict[str, Tensor]],
    ) -> Dict[str, float]:
        """Single Reptile meta-training step.
        
        Args:
            task_batch: List of tasks with support sets
            
        Returns:
            metrics: Training metrics
        """
        # Store initial parameters
        initial_params = {
            name: param.clone()
            for name, param in self.model.named_parameters()
        }
        
        total_loss = 0.0
        
        for task in task_batch:
            support_ids = task["support_ids"]
            support_targets = task["support_targets"]
            
            # Reset to initial parameters
            for name, param in self.model.named_parameters():
                param.data.copy_(initial_params[name])
            
            # Adapt on support set
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.inner_lr,
            )
            
            for _ in range(self.config.num_inner_steps):
                out = self.model(support_ids, targets=support_targets)
                loss = out["loss"]
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Interpolate towards adapted parameters
            for name, param in self.model.named_parameters():
                initial_params[name] += self.meta_lr * (param.data - initial_params[name])
        
        # Update model with interpolated parameters
        for name, param in self.model.named_parameters():
            param.data.copy_(initial_params[name])
        
        return {"meta_loss": total_loss / (len(task_batch) * self.config.num_inner_steps)}


class WaveFieldMetaLearner:
    """Complete meta-learning system for Wave Field LLM.
    
    Supports multiple meta-learning algorithms and few-shot scenarios.
    """
    
    def __init__(
        self,
        model_config: WaveFieldConfig,
        meta_config: MetaLearningConfig,
        algorithm: str = "maml",  # maml | prototypical | matching | reptile
    ):
        self.model_config = model_config
        self.meta_config = meta_config
        self.algorithm = algorithm
        
        # Initialize model
        self.model = WaveFieldLM(model_config)
        
        # Initialize meta-learner
        if algorithm == "maml":
            self.learner = MAMLOptimizer(self.model, meta_config)
        elif algorithm == "prototypical":
            self.learner = PrototypicalNetwork(self.model)
        elif algorithm == "matching":
            self.learner = MatchingNetwork(self.model)
        elif algorithm == "reptile":
            self.learner = ReptileOptimizer(self.model, meta_config)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def meta_train(
        self,
        task_distribution,
        num_iterations: int = 1000,
    ) -> List[Dict[str, float]]:
        """Meta-train on a distribution of tasks.
        
        Args:
            task_distribution: Iterator that yields task batches
            num_iterations: Number of meta-training iterations
            
        Returns:
            metrics_history: List of metrics per iteration
        """
        metrics_history = []
        
        for iteration in range(num_iterations):
            # Sample task batch
            task_batch = next(task_distribution)
            
            # Meta-training step
            if self.algorithm in ["maml", "reptile"]:
                metrics = self.learner.meta_train_step(task_batch)
            else:
                # For prototypical/matching, train with episodic batches
                metrics = self._train_episodic(task_batch)
            
            metrics_history.append(metrics)
            
            if (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{num_iterations}, "
                      f"loss: {metrics.get('meta_loss', 0.0):.4f}")
        
        return metrics_history
    
    def _train_episodic(self, task_batch: List[Dict[str, Tensor]]) -> Dict[str, float]:
        """Train with episodic batches for prototypical/matching networks."""
        total_loss = 0.0
        optimizer = torch.optim.Adam(self.learner.parameters(), lr=self.meta_config.outer_lr)
        
        for task in task_batch:
            support_ids = task["support_ids"]
            support_labels = task["support_labels"]
            query_ids = task["query_ids"]
            query_labels = task["query_labels"]
            
            if self.algorithm == "prototypical":
                # Compute prototypes
                prototypes = self.learner.compute_prototypes(
                    support_ids, support_labels, self.meta_config.num_ways
                )
                # Classify queries
                logits = self.learner(query_ids, prototypes)
            else:  # matching
                logits = self.learner(
                    query_ids, support_ids, support_labels, self.meta_config.num_ways
                )
            
            # Compute loss
            loss = F.cross_entropy(logits, query_labels)
            total_loss += loss.item()
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return {"meta_loss": total_loss / len(task_batch)}
    
    def few_shot_adapt(
        self,
        support_ids: Tensor,
        support_targets: Tensor,
    ) -> WaveFieldLM:
        """Adapt to a new task with few examples.
        
        Args:
            support_ids: [K, N] support examples
            support_targets: [K, N] support targets
            
        Returns:
            adapted_model: Model adapted to the task
        """
        if self.algorithm == "maml":
            return self.learner.adapt_to_task(support_ids, support_targets)
        elif self.algorithm == "reptile":
            # Reptile adaptation is same as MAML inner loop
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.meta_config.inner_lr,
            )
            for _ in range(self.meta_config.num_inner_steps):
                out = self.model(support_ids, targets=support_targets)
                loss = out["loss"]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            return self.model
        else:
            # Prototypical/matching don't need adaptation
            return self.model

# Made with Bob
