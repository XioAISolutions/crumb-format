"""Advanced research features for Wave Field LLM.

This module contains cutting-edge research implementations that push the
boundaries of wave-based language models:

- MultiModalWaveField: Multi-modal learning (text, images, audio, code)
- WaveFieldRLHF: Reinforcement learning from human feedback
- ContinualWaveField: Lifelong learning without catastrophic forgetting
- WaveFieldNAS: Neural architecture search for optimal wave configurations
- WaveFieldAnalyzer: Interpretability and analysis tools
- WaveFieldMetaLearner: Few-shot and meta-learning
- SparseWaveField: Efficient sparse architectures
- LongContextWaveField: Extended context handling (100K+ tokens)
"""

from .multimodal import MultiModalWaveField, ModalityEncoder
from .rlhf import WaveFieldRLHF, RewardModel, PPOTrainer
from .continual import ContinualWaveField, EWCRegularizer, TaskMemory
from .nas import WaveFieldNAS, SearchSpace, ArchitectureEvaluator
from .interpretability import WaveFieldAnalyzer, WaveVisualizer, ConceptDiscovery
from .meta_learning import WaveFieldMetaLearner, MAMLOptimizer, PrototypicalNetwork
from .sparse import SparseWaveField, MixtureOfExperts, WaveRouter
from .long_context import LongContextWaveField, HierarchicalField, CompressiveMemory

__all__ = [
    # Multi-modal
    "MultiModalWaveField",
    "ModalityEncoder",
    # RLHF
    "WaveFieldRLHF",
    "RewardModel",
    "PPOTrainer",
    # Continual learning
    "ContinualWaveField",
    "EWCRegularizer",
    "TaskMemory",
    # NAS
    "WaveFieldNAS",
    "SearchSpace",
    "ArchitectureEvaluator",
    # Interpretability
    "WaveFieldAnalyzer",
    "WaveVisualizer",
    "ConceptDiscovery",
    # Meta-learning
    "WaveFieldMetaLearner",
    "MAMLOptimizer",
    "PrototypicalNetwork",
    # Sparse
    "SparseWaveField",
    "MixtureOfExperts",
    "WaveRouter",
    # Long context
    "LongContextWaveField",
    "HierarchicalField",
    "CompressiveMemory",
]

# Made with Bob
