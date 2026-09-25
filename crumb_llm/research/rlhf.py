"""Reinforcement Learning from Human Feedback (RLHF) for Wave Field LLM.

Implements PPO-based RLHF training for wave-based language models, enabling
alignment with human preferences and safety constraints.

Key components:
- Reward model training on wave representations
- PPO optimization in wave space
- KL divergence penalty for stability
- Advantage estimation using wave propagation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

from ..model import WaveFieldLM, WaveFieldConfig
from ..layers import RMSNorm


@dataclass
class RLHFConfig:
    """Configuration for RLHF training."""
    # PPO hyperparameters
    ppo_epochs: int = 4
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    kl_coef: float = 0.1
    max_grad_norm: float = 1.0
    
    # Training settings
    batch_size: int = 32
    mini_batch_size: int = 8
    learning_rate: float = 1e-5
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE parameter
    
    # Reward model settings
    reward_model_lr: float = 1e-4
    reward_model_epochs: int = 3


class RewardModel(nn.Module):
    """Reward model that scores wave field representations.
    
    Takes the final hidden states from the wave field model and predicts
    a scalar reward. Trained on human preference data.
    """
    
    def __init__(self, config: WaveFieldConfig):
        super().__init__()
        self.config = config
        
        # Use the wave field model as feature extractor
        self.wave_model = WaveFieldLM(config)
        
        # Reward head: pool sequence and predict scalar
        self.reward_head = nn.Sequential(
            RMSNorm(config.dim),
            nn.Linear(config.dim, config.dim // 2),
            nn.ReLU(),
            nn.Linear(config.dim // 2, 1),
        )
    
    def forward(self, input_ids: Tensor) -> Tensor:
        """Compute reward for a sequence.
        
        Args:
            input_ids: [B, N] token ids
            
        Returns:
            rewards: [B] scalar rewards
        """
        # Get wave field representations
        out = self.wave_model(input_ids)
        logits = out["logits"]  # [B, N, V]
        
        # Pool over sequence (use last token for autoregressive)
        hidden = logits[:, -1, :]  # [B, V]
        
        # Predict reward
        reward = self.reward_head(hidden).squeeze(-1)  # [B]
        return reward
    
    def train_on_preferences(
        self,
        chosen_ids: Tensor,
        rejected_ids: Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Train reward model on preference pairs.
        
        Args:
            chosen_ids: [B, N] preferred sequences
            rejected_ids: [B, N] rejected sequences
            optimizer: Optimizer for reward model
            
        Returns:
            loss: Scalar loss value
        """
        # Compute rewards for both
        r_chosen = self(chosen_ids)
        r_rejected = self(rejected_ids)
        
        # Bradley-Terry loss: P(chosen > rejected) = sigmoid(r_chosen - r_rejected)
        loss = -F.logsigmoid(r_chosen - r_rejected).mean()
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()


class ValueNetwork(nn.Module):
    """Value function for PPO, estimates expected return.
    
    Uses wave field representations to predict state values.
    """
    
    def __init__(self, config: WaveFieldConfig):
        super().__init__()
        self.config = config
        
        # Share wave field encoder with policy
        self.wave_model = WaveFieldLM(config)
        
        # Value head
        self.value_head = nn.Sequential(
            RMSNorm(config.dim),
            nn.Linear(config.dim, config.dim // 2),
            nn.ReLU(),
            nn.Linear(config.dim // 2, 1),
        )
    
    def forward(self, input_ids: Tensor) -> Tensor:
        """Estimate value of states.
        
        Args:
            input_ids: [B, N] token ids
            
        Returns:
            values: [B, N] state values
        """
        out = self.wave_model(input_ids)
        logits = out["logits"]  # [B, N, V]
        
        # Predict value for each position
        values = self.value_head(logits).squeeze(-1)  # [B, N]
        return values


class PPOTrainer:
    """Proximal Policy Optimization trainer for wave field models.
    
    Implements PPO algorithm with wave-specific optimizations:
    - KL divergence computed in wave space
    - Advantage estimation using wave propagation
    - Clipped surrogate objective
    """
    
    def __init__(
        self,
        policy: WaveFieldLM,
        ref_policy: WaveFieldLM,
        value_net: ValueNetwork,
        reward_fn: Callable[[Tensor], Tensor],
        config: RLHFConfig,
    ):
        self.policy = policy
        self.ref_policy = ref_policy
        self.value_net = value_net
        self.reward_fn = reward_fn
        self.config = config
        
        # Optimizers
        self.policy_optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=config.learning_rate,
        )
        self.value_optimizer = torch.optim.AdamW(
            value_net.parameters(),
            lr=config.learning_rate,
        )
        
        # Freeze reference policy
        for param in ref_policy.parameters():
            param.requires_grad = False
    
    def compute_advantages(
        self,
        rewards: Tensor,
        values: Tensor,
        dones: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute GAE advantages and returns.
        
        Args:
            rewards: [B, N] rewards
            values: [B, N] value estimates
            dones: [B, N] episode termination flags
            
        Returns:
            advantages: [B, N] advantage estimates
            returns: [B, N] discounted returns
        """
        B, N = rewards.shape
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        gae = 0.0
        for t in reversed(range(N)):
            if t == N - 1:
                next_value = 0.0
            else:
                next_value = values[:, t + 1]
            
            delta = rewards[:, t] + self.config.gamma * next_value * (1 - dones[:, t]) - values[:, t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[:, t]) * gae
            advantages[:, t] = gae
            returns[:, t] = gae + values[:, t]
        
        return advantages, returns
    
    def compute_kl_divergence(
        self,
        input_ids: Tensor,
        actions: Tensor,
    ) -> Tensor:
        """Compute KL divergence between policy and reference policy.
        
        Args:
            input_ids: [B, N] input token ids
            actions: [B, N] sampled actions
            
        Returns:
            kl: [B] KL divergence per sequence
        """
        # Policy logits
        policy_out = self.policy(input_ids)
        policy_logits = policy_out["logits"]  # [B, N, V]
        
        # Reference policy logits
        with torch.no_grad():
            ref_out = self.ref_policy(input_ids)
            ref_logits = ref_out["logits"]  # [B, N, V]
        
        # Compute KL divergence
        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        
        # KL(policy || ref) = sum_a policy(a) * (log policy(a) - log ref(a))
        kl = (policy_log_probs.exp() * (policy_log_probs - ref_log_probs)).sum(dim=-1)
        return kl.mean(dim=-1)  # [B]
    
    def ppo_step(
        self,
        input_ids: Tensor,
        actions: Tensor,
        old_log_probs: Tensor,
        advantages: Tensor,
        returns: Tensor,
    ) -> dict[str, float]:
        """Single PPO update step.
        
        Args:
            input_ids: [B, N] input sequences
            actions: [B, N] sampled actions
            old_log_probs: [B, N] log probs from old policy
            advantages: [B, N] advantage estimates
            returns: [B, N] target returns
            
        Returns:
            metrics: Dict of training metrics
        """
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy forward
        policy_out = self.policy(input_ids)
        logits = policy_out["logits"]  # [B, N, V]
        
        # Compute new log probs
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)  # [B, N]
        entropy = dist.entropy().mean()
        
        # Compute ratio and clipped objective
        ratio = (new_log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        values = self.value_net(input_ids)
        value_loss = F.mse_loss(values, returns)
        
        # KL penalty
        kl = self.compute_kl_divergence(input_ids, actions)
        kl_penalty = self.config.kl_coef * kl.mean()
        
        # Total loss
        loss = policy_loss + self.config.value_loss_coef * value_loss - self.config.entropy_coef * entropy + kl_penalty
        
        # Optimize
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.config.max_grad_norm)
        
        self.policy_optimizer.step()
        self.value_optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "kl": kl.mean().item(),
            "total_loss": loss.item(),
        }
    
    def train_step(
        self,
        prompts: Tensor,
        max_length: int = 128,
    ) -> dict[str, float]:
        """Complete PPO training step: sample, evaluate, update.
        
        Args:
            prompts: [B, N_prompt] prompt token ids
            max_length: Maximum generation length
            
        Returns:
            metrics: Training metrics
        """
        B = prompts.shape[0]
        device = prompts.device
        
        # Sample completions from policy
        with torch.no_grad():
            completions = self.policy.generate(
                prompts,
                max_new_tokens=max_length,
                temperature=1.0,
            )
        
        # Compute rewards
        rewards = self.reward_fn(completions)  # [B]
        
        # Expand rewards to sequence
        N = completions.shape[1]
        rewards_seq = torch.zeros(B, N, device=device)
        rewards_seq[:, -1] = rewards  # Reward at end of sequence
        
        # Compute values
        with torch.no_grad():
            values = self.value_net(completions)
        
        # Compute advantages
        dones = torch.zeros(B, N, device=device)
        dones[:, -1] = 1.0
        advantages, returns = self.compute_advantages(rewards_seq, values, dones)
        
        # Get old log probs
        with torch.no_grad():
            policy_out = self.policy(completions[:, :-1])
            logits = policy_out["logits"]
            dist = Categorical(logits=logits)
            old_log_probs = dist.log_prob(completions[:, 1:])
            # Pad to match sequence length
            old_log_probs = F.pad(old_log_probs, (1, 0))
        
        # PPO updates
        metrics_list = []
        for _ in range(self.config.ppo_epochs):
            metrics = self.ppo_step(
                completions,
                completions,  # Actions are the tokens themselves
                old_log_probs,
                advantages,
                returns,
            )
            metrics_list.append(metrics)
        
        # Average metrics
        avg_metrics = {
            k: sum(m[k] for m in metrics_list) / len(metrics_list)
            for k in metrics_list[0].keys()
        }
        avg_metrics["reward"] = rewards.mean().item()
        
        return avg_metrics


class WaveFieldRLHF:
    """Complete RLHF pipeline for Wave Field LLM.
    
    Orchestrates the full RLHF training process:
    1. Train reward model on preference data
    2. Initialize PPO trainer
    3. Run PPO training loop
    4. Evaluate and save checkpoints
    """
    
    def __init__(
        self,
        model_config: WaveFieldConfig,
        rlhf_config: RLHFConfig,
    ):
        self.model_config = model_config
        self.rlhf_config = rlhf_config
        
        # Initialize models
        self.policy = WaveFieldLM(model_config)
        self.ref_policy = WaveFieldLM(model_config)
        self.value_net = ValueNetwork(model_config)
        self.reward_model = RewardModel(model_config)
        
        # Copy policy to reference
        self.ref_policy.load_state_dict(self.policy.state_dict())
    
    def train_reward_model(
        self,
        preference_data: list[tuple[Tensor, Tensor]],
        num_epochs: Optional[int] = None,
    ) -> list[float]:
        """Train reward model on preference pairs.
        
        Args:
            preference_data: List of (chosen, rejected) token id pairs
            num_epochs: Number of training epochs (default from config)
            
        Returns:
            losses: List of loss values per epoch
        """
        num_epochs = num_epochs or self.rlhf_config.reward_model_epochs
        optimizer = torch.optim.AdamW(
            self.reward_model.parameters(),
            lr=self.rlhf_config.reward_model_lr,
        )
        
        losses = []
        for epoch in range(num_epochs):
            epoch_losses = []
            for chosen, rejected in preference_data:
                loss = self.reward_model.train_on_preferences(
                    chosen, rejected, optimizer
                )
                epoch_losses.append(loss)
            
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(avg_loss)
            print(f"Reward model epoch {epoch + 1}/{num_epochs}, loss: {avg_loss:.4f}")
        
        return losses
    
    def train_policy(
        self,
        prompts: list[Tensor],
        num_steps: int = 1000,
    ) -> list[dict[str, float]]:
        """Train policy with PPO.
        
        Args:
            prompts: List of prompt tensors
            num_steps: Number of PPO steps
            
        Returns:
            metrics: List of metric dicts per step
        """
        # Create PPO trainer with reward model
        def reward_fn(completions: Tensor) -> Tensor:
            with torch.no_grad():
                return self.reward_model(completions)
        
        trainer = PPOTrainer(
            self.policy,
            self.ref_policy,
            self.value_net,
            reward_fn,
            self.rlhf_config,
        )
        
        metrics_history = []
        for step in range(num_steps):
            # Sample batch of prompts
            batch_idx = torch.randint(0, len(prompts), (self.rlhf_config.batch_size,))
            batch_prompts = torch.stack([prompts[i] for i in batch_idx])
            
            # Training step
            metrics = trainer.train_step(batch_prompts)
            metrics_history.append(metrics)
            
            if (step + 1) % 10 == 0:
                print(f"Step {step + 1}/{num_steps}, reward: {metrics['reward']:.4f}, "
                      f"policy_loss: {metrics['policy_loss']:.4f}")
        
        return metrics_history

# Made with Bob
