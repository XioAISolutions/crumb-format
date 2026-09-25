"""Neural Architecture Search for Wave Field LLM.

Automated discovery of optimal wave field architectures through:
- DARTS-style differentiable search
- Evolutionary search with wave fitness
- Hardware-aware optimization
- Multi-objective search (accuracy, speed, size)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..model import WaveFieldConfig, WaveFieldLM
from ..layers import WaveFieldBlock, WaveFieldBlockConfig


@dataclass
class SearchSpace:
    """Defines the architecture search space."""
    n_heads_choices: List[int] = None
    field_size_choices: List[int] = None
    n_layers_choices: List[int] = None
    dim_choices: List[int] = None
    alpha_range: Tuple[float, float] = (0.01, 0.2)
    omega_range: Tuple[float, float] = (0.5, 4.0)
    
    def __post_init__(self):
        if self.n_heads_choices is None:
            self.n_heads_choices = [2, 4, 8, 16]
        if self.field_size_choices is None:
            self.field_size_choices = [128, 256, 512, 1024]
        if self.n_layers_choices is None:
            self.n_layers_choices = [4, 6, 8, 12]
        if self.dim_choices is None:
            self.dim_choices = [128, 256, 512, 768]


@dataclass
class NASConfig:
    """Configuration for NAS."""
    search_space: SearchSpace
    search_method: str = "darts"  # darts | evolutionary | random
    num_epochs: int = 50
    population_size: int = 20  # For evolutionary
    mutation_rate: float = 0.1
    hardware_constraint: Optional[str] = None  # latency | memory | flops
    multi_objective: bool = True


class ArchitectureEvaluator:
    """Evaluates architecture candidates on multiple metrics."""
    
    def __init__(self, val_data_loader, device: str = "cuda"):
        self.val_data_loader = val_data_loader
        self.device = device
    
    def evaluate(self, model: WaveFieldLM) -> Dict[str, float]:
        """Evaluate model on validation set.
        
        Args:
            model: Model to evaluate
            
        Returns:
            metrics: Dict with accuracy, loss, latency, memory, params
        """
        model.eval()
        model.to(self.device)
        
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        
        # Measure latency
        import time
        start_time = time.time()
        
        with torch.no_grad():
            for batch in self.val_data_loader:
                input_ids = batch["input_ids"].to(self.device)
                targets = batch.get("targets", input_ids)
                
                out = model(input_ids, targets=targets)
                loss = out["loss"]
                logits = out["logits"]
                
                total_loss += loss.item() * input_ids.shape[0]
                preds = logits.argmax(dim=-1)
                total_correct += (preds == targets).sum().item()
                total_tokens += targets.numel()
        
        latency = (time.time() - start_time) / len(self.val_data_loader)
        
        # Measure memory
        memory_mb = torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
        
        # Count parameters
        num_params = model.num_parameters()
        
        metrics = {
            "accuracy": total_correct / total_tokens if total_tokens > 0 else 0.0,
            "loss": total_loss / len(self.val_data_loader.dataset),
            "latency_ms": latency * 1000,
            "memory_mb": memory_mb,
            "params_m": num_params / 1e6,
        }
        
        model.train()
        return metrics
    
    def compute_fitness(
        self,
        metrics: Dict[str, float],
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """Compute multi-objective fitness score.
        
        Args:
            metrics: Evaluation metrics
            weights: Weights for each metric (default: equal)
            
        Returns:
            fitness: Scalar fitness score (higher is better)
        """
        if weights is None:
            weights = {
                "accuracy": 1.0,
                "loss": -0.5,  # Negative because lower is better
                "latency_ms": -0.001,
                "memory_mb": -0.0001,
                "params_m": -0.01,
            }
        
        fitness = 0.0
        for key, weight in weights.items():
            if key in metrics:
                fitness += weight * metrics[key]
        
        return fitness


class DARTSSearcher:
    """Differentiable Architecture Search for wave fields.
    
    Uses continuous relaxation of architecture choices with Gumbel-Softmax.
    """
    
    def __init__(
        self,
        search_space: SearchSpace,
        base_config: WaveFieldConfig,
    ):
        self.search_space = search_space
        self.base_config = base_config
        
        # Architecture parameters (continuous relaxation)
        self.arch_params = nn.ParameterDict({
            "n_heads": nn.Parameter(torch.randn(len(search_space.n_heads_choices))),
            "field_size": nn.Parameter(torch.randn(len(search_space.field_size_choices))),
            "n_layers": nn.Parameter(torch.randn(len(search_space.n_layers_choices))),
            "dim": nn.Parameter(torch.randn(len(search_space.dim_choices))),
        })
    
    def sample_architecture(self, temperature: float = 1.0) -> WaveFieldConfig:
        """Sample architecture using Gumbel-Softmax.
        
        Args:
            temperature: Gumbel-Softmax temperature
            
        Returns:
            config: Sampled architecture configuration
        """
        # Sample discrete choices with Gumbel-Softmax
        n_heads_probs = F.gumbel_softmax(self.arch_params["n_heads"], tau=temperature, hard=True)
        n_heads = sum(p * c for p, c in zip(n_heads_probs, self.search_space.n_heads_choices))
        
        field_size_probs = F.gumbel_softmax(self.arch_params["field_size"], tau=temperature, hard=True)
        field_size = sum(p * c for p, c in zip(field_size_probs, self.search_space.field_size_choices))
        
        n_layers_probs = F.gumbel_softmax(self.arch_params["n_layers"], tau=temperature, hard=True)
        n_layers = sum(p * c for p, c in zip(n_layers_probs, self.search_space.n_layers_choices))
        
        dim_probs = F.gumbel_softmax(self.arch_params["dim"], tau=temperature, hard=True)
        dim = sum(p * c for p, c in zip(dim_probs, self.search_space.dim_choices))
        
        config = WaveFieldConfig(
            vocab_size=self.base_config.vocab_size,
            dim=int(dim.item()),
            n_layers=int(n_layers.item()),
            n_heads=int(n_heads.item()),
            field_size=int(field_size.item()),
        )
        
        return config
    
    def get_best_architecture(self) -> WaveFieldConfig:
        """Get the best architecture from learned parameters.
        
        Returns:
            config: Best architecture configuration
        """
        # Take argmax of architecture parameters
        n_heads_idx = self.arch_params["n_heads"].argmax().item()
        field_size_idx = self.arch_params["field_size"].argmax().item()
        n_layers_idx = self.arch_params["n_layers"].argmax().item()
        dim_idx = self.arch_params["dim"].argmax().item()
        
        config = WaveFieldConfig(
            vocab_size=self.base_config.vocab_size,
            dim=self.search_space.dim_choices[dim_idx],
            n_layers=self.search_space.n_layers_choices[n_layers_idx],
            n_heads=self.search_space.n_heads_choices[n_heads_idx],
            field_size=self.search_space.field_size_choices[field_size_idx],
        )
        
        return config


class EvolutionarySearcher:
    """Evolutionary search for wave field architectures.
    
    Uses genetic algorithm with mutation and crossover.
    """
    
    def __init__(
        self,
        search_space: SearchSpace,
        base_config: WaveFieldConfig,
        population_size: int = 20,
        mutation_rate: float = 0.1,
    ):
        self.search_space = search_space
        self.base_config = base_config
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        
        # Initialize population
        self.population: List[WaveFieldConfig] = []
        self.fitness_scores: List[float] = []
    
    def initialize_population(self) -> None:
        """Initialize random population."""
        self.population = []
        for _ in range(self.population_size):
            config = WaveFieldConfig(
                vocab_size=self.base_config.vocab_size,
                dim=torch.randint(0, len(self.search_space.dim_choices), (1,)).item(),
                n_layers=torch.randint(0, len(self.search_space.n_layers_choices), (1,)).item(),
                n_heads=torch.randint(0, len(self.search_space.n_heads_choices), (1,)).item(),
                field_size=torch.randint(0, len(self.search_space.field_size_choices), (1,)).item(),
            )
            # Map indices to actual values
            config.dim = self.search_space.dim_choices[config.dim]
            config.n_layers = self.search_space.n_layers_choices[config.n_layers]
            config.n_heads = self.search_space.n_heads_choices[config.n_heads]
            config.field_size = self.search_space.field_size_choices[config.field_size]
            
            self.population.append(config)
    
    def mutate(self, config: WaveFieldConfig) -> WaveFieldConfig:
        """Mutate an architecture.
        
        Args:
            config: Configuration to mutate
            
        Returns:
            mutated_config: Mutated configuration
        """
        new_config = WaveFieldConfig(**config.__dict__)
        
        if torch.rand(1).item() < self.mutation_rate:
            new_config.n_heads = torch.tensor(self.search_space.n_heads_choices).random_().item()
        
        if torch.rand(1).item() < self.mutation_rate:
            new_config.field_size = torch.tensor(self.search_space.field_size_choices).random_().item()
        
        if torch.rand(1).item() < self.mutation_rate:
            new_config.n_layers = torch.tensor(self.search_space.n_layers_choices).random_().item()
        
        if torch.rand(1).item() < self.mutation_rate:
            new_config.dim = torch.tensor(self.search_space.dim_choices).random_().item()
        
        return new_config
    
    def crossover(
        self,
        config1: WaveFieldConfig,
        config2: WaveFieldConfig,
    ) -> WaveFieldConfig:
        """Crossover two architectures.
        
        Args:
            config1: First parent
            config2: Second parent
            
        Returns:
            child_config: Child configuration
        """
        child_config = WaveFieldConfig(
            vocab_size=self.base_config.vocab_size,
            dim=config1.dim if torch.rand(1).item() < 0.5 else config2.dim,
            n_layers=config1.n_layers if torch.rand(1).item() < 0.5 else config2.n_layers,
            n_heads=config1.n_heads if torch.rand(1).item() < 0.5 else config2.n_heads,
            field_size=config1.field_size if torch.rand(1).item() < 0.5 else config2.field_size,
        )
        
        return child_config
    
    def evolve(self, evaluator: ArchitectureEvaluator) -> None:
        """Perform one generation of evolution.
        
        Args:
            evaluator: Architecture evaluator
        """
        # Evaluate population
        self.fitness_scores = []
        for config in self.population:
            model = WaveFieldLM(config)
            metrics = evaluator.evaluate(model)
            fitness = evaluator.compute_fitness(metrics)
            self.fitness_scores.append(fitness)
        
        # Selection: keep top 50%
        sorted_indices = sorted(range(len(self.fitness_scores)), 
                              key=lambda i: self.fitness_scores[i], 
                              reverse=True)
        elite_size = self.population_size // 2
        elite_indices = sorted_indices[:elite_size]
        elite = [self.population[i] for i in elite_indices]
        
        # Generate new population
        new_population = elite.copy()
        
        while len(new_population) < self.population_size:
            # Select two parents
            parent1 = elite[torch.randint(0, len(elite), (1,)).item()]
            parent2 = elite[torch.randint(0, len(elite), (1,)).item()]
            
            # Crossover
            child = self.crossover(parent1, parent2)
            
            # Mutate
            child = self.mutate(child)
            
            new_population.append(child)
        
        self.population = new_population
    
    def get_best_architecture(self) -> WaveFieldConfig:
        """Get the best architecture from population.
        
        Returns:
            config: Best architecture
        """
        best_idx = max(range(len(self.fitness_scores)), 
                      key=lambda i: self.fitness_scores[i])
        return self.population[best_idx]


class WaveFieldNAS:
    """Complete Neural Architecture Search system for Wave Field LLM.
    
    Orchestrates architecture search using various methods.
    """
    
    def __init__(
        self,
        base_config: WaveFieldConfig,
        nas_config: NASConfig,
    ):
        self.base_config = base_config
        self.nas_config = nas_config
        
        # Initialize searcher
        if nas_config.search_method == "darts":
            self.searcher = DARTSSearcher(nas_config.search_space, base_config)
        elif nas_config.search_method == "evolutionary":
            self.searcher = EvolutionarySearcher(
                nas_config.search_space,
                base_config,
                nas_config.population_size,
                nas_config.mutation_rate,
            )
        else:
            raise ValueError(f"Unknown search method: {nas_config.search_method}")
    
    def search(
        self,
        train_data_loader,
        val_data_loader,
        device: str = "cuda",
    ) -> WaveFieldConfig:
        """Run architecture search.
        
        Args:
            train_data_loader: Training data
            val_data_loader: Validation data
            device: Device to use
            
        Returns:
            best_config: Best found architecture
        """
        evaluator = ArchitectureEvaluator(val_data_loader, device)
        
        if self.nas_config.search_method == "darts":
            return self._search_darts(train_data_loader, evaluator, device)
        elif self.nas_config.search_method == "evolutionary":
            return self._search_evolutionary(evaluator)
        
        raise ValueError(f"Unknown search method: {self.nas_config.search_method}")
    
    def _search_darts(
        self,
        train_data_loader,
        evaluator: ArchitectureEvaluator,
        device: str,
    ) -> WaveFieldConfig:
        """DARTS search implementation."""
        optimizer = torch.optim.Adam(
            self.searcher.arch_params.parameters(),
            lr=3e-4,
        )
        
        for epoch in range(self.nas_config.num_epochs):
            # Sample architecture
            temperature = max(0.1, 1.0 - epoch / self.nas_config.num_epochs)
            config = self.searcher.sample_architecture(temperature)
            
            # Train sampled architecture briefly
            model = WaveFieldLM(config).to(device)
            model_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            
            for batch in train_data_loader:
                input_ids = batch["input_ids"].to(device)
                targets = batch.get("targets", input_ids)
                
                out = model(input_ids, targets=targets)
                loss = out["loss"]
                
                model_optimizer.zero_grad()
                loss.backward()
                model_optimizer.step()
            
            # Evaluate
            metrics = evaluator.evaluate(model)
            fitness = evaluator.compute_fitness(metrics)
            
            # Update architecture parameters (maximize fitness)
            arch_loss = -fitness
            optimizer.zero_grad()
            arch_loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch + 1}/{self.nas_config.num_epochs}, "
                  f"fitness: {fitness:.4f}, "
                  f"accuracy: {metrics['accuracy']:.4f}")
        
        return self.searcher.get_best_architecture()
    
    def _search_evolutionary(
        self,
        evaluator: ArchitectureEvaluator,
    ) -> WaveFieldConfig:
        """Evolutionary search implementation."""
        self.searcher.initialize_population()
        
        for generation in range(self.nas_config.num_epochs):
            self.searcher.evolve(evaluator)
            
            best_fitness = max(self.searcher.fitness_scores)
            avg_fitness = sum(self.searcher.fitness_scores) / len(self.searcher.fitness_scores)
            
            print(f"Generation {generation + 1}/{self.nas_config.num_epochs}, "
                  f"best_fitness: {best_fitness:.4f}, "
                  f"avg_fitness: {avg_fitness:.4f}")
        
        return self.searcher.get_best_architecture()

# Made with Bob
