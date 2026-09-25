"""Comprehensive tests for Wave Field LLM research features."""

import pytest
import torch
import torch.nn as nn

from crumb_llm.model import WaveFieldConfig
from crumb_llm.research import (
    MultiModalWaveField, MultiModalWaveFieldConfig, ModalityConfig,
    WaveFieldRLHF, RLHFConfig, RewardModel,
    ContinualWaveField, ContinualConfig,
    WaveFieldNAS, NASConfig, SearchSpace,
    WaveFieldAnalyzer, AnalysisConfig,
    WaveFieldMetaLearner, MetaLearningConfig,
    SparseWaveField, SparseConfig,
    LongContextWaveField, LongContextConfig,
)


@pytest.fixture
def base_config():
    """Base configuration for testing."""
    return WaveFieldConfig(
        vocab_size=256,
        dim=64,
        n_layers=2,
        n_heads=4,
        field_size=128,
    )


@pytest.fixture
def device():
    """Test device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


class TestMultiModal:
    """Tests for multi-modal wave fields."""
    
    def test_multimodal_creation(self, base_config):
        """Test creating multi-modal model."""
        modalities = [
            ModalityConfig(name="text", vocab_size=256),
            ModalityConfig(name="image", vocab_size=0, patch_size=16),
        ]
        config = MultiModalWaveFieldConfig(
            base_config=base_config,
            modalities=modalities,
        )
        model = MultiModalWaveField(config)
        assert model is not None
    
    def test_multimodal_forward(self, base_config):
        """Test multi-modal forward pass."""
        modalities = [
            ModalityConfig(name="text", vocab_size=256),
        ]
        config = MultiModalWaveFieldConfig(
            base_config=base_config,
            modalities=modalities,
        )
        model = MultiModalWaveField(config)
        
        # Test forward
        inputs = {"text": torch.randint(0, 256, (2, 10))}
        out = model(inputs)
        
        assert "logits" in out
        assert "text" in out["logits"]
        assert out["logits"]["text"].shape == (2, 10, 256)
    
    def test_cross_modal_interference(self, base_config):
        """Test cross-modal interference."""
        from crumb_llm.research.multimodal import CrossModalInterference
        
        interference = CrossModalInterference(2, base_config.dim, base_config.field_size)
        
        # Create two fields
        field1 = torch.randn(2, 4, 128, 16)
        field2 = torch.randn(2, 4, 128, 16)
        
        interfered = interference([field1, field2])
        
        assert len(interfered) == 2
        assert interfered[0].shape == field1.shape


class TestRLHF:
    """Tests for RLHF."""
    
    def test_reward_model(self, base_config):
        """Test reward model creation and forward."""
        model = RewardModel(base_config)
        
        input_ids = torch.randint(0, 256, (2, 10))
        rewards = model(input_ids)
        
        assert rewards.shape == (2,)
    
    def test_rlhf_system(self, base_config):
        """Test complete RLHF system."""
        rlhf_config = RLHFConfig(
            ppo_epochs=1,
            batch_size=2,
        )
        rlhf = WaveFieldRLHF(base_config, rlhf_config)
        
        assert rlhf.policy is not None
        assert rlhf.reward_model is not None


class TestContinualLearning:
    """Tests for continual learning."""
    
    def test_ewc_regularizer(self, base_config):
        """Test EWC regularizer."""
        from crumb_llm.research.continual import EWCRegularizer
        from crumb_llm.model import WaveFieldLM
        
        model = WaveFieldLM(base_config)
        ewc = EWCRegularizer(model, lambda_=1000.0)
        
        # Test penalty computation (should be zero initially)
        penalty = ewc.penalty()
        assert penalty.item() == 0.0
    
    def test_task_memory(self):
        """Test task memory buffer."""
        from crumb_llm.research.continual import TaskMemory
        
        memory = TaskMemory(max_size=100)
        
        # Add task data
        data = [
            {"input_ids": torch.randint(0, 256, (10,)), "targets": torch.randint(0, 256, (10,))}
            for _ in range(50)
        ]
        memory.add_task_data(data)
        
        assert len(memory.buffer) == 50
        
        # Sample
        batch = memory.sample(10)
        assert len(batch) == 10
    
    def test_continual_system(self, base_config):
        """Test continual learning system."""
        continual_config = ContinualConfig(memory_size=100)
        system = ContinualWaveField(base_config, continual_config)
        
        assert system.model is not None
        assert system.ewc is not None


class TestNAS:
    """Tests for neural architecture search."""
    
    def test_search_space(self):
        """Test search space definition."""
        space = SearchSpace(
            n_heads_choices=[2, 4, 8],
            field_size_choices=[128, 256],
        )
        assert len(space.n_heads_choices) == 3
        assert len(space.field_size_choices) == 2
    
    def test_darts_searcher(self, base_config):
        """Test DARTS searcher."""
        from crumb_llm.research.nas import DARTSSearcher
        
        space = SearchSpace()
        searcher = DARTSSearcher(space, base_config)
        
        # Sample architecture
        config = searcher.sample_architecture(temperature=1.0)
        assert config.dim in space.dim_choices
    
    def test_evolutionary_searcher(self, base_config):
        """Test evolutionary searcher."""
        from crumb_llm.research.nas import EvolutionarySearcher
        
        space = SearchSpace()
        searcher = EvolutionarySearcher(space, base_config, population_size=5)
        
        searcher.initialize_population()
        assert len(searcher.population) == 5


class TestInterpretability:
    """Tests for interpretability tools."""
    
    def test_wave_visualizer(self, base_config):
        """Test wave visualizer."""
        from crumb_llm.model import WaveFieldLM
        from crumb_llm.research.interpretability import WaveVisualizer
        
        model = WaveFieldLM(base_config)
        visualizer = WaveVisualizer(model)
        
        input_ids = torch.randint(0, 256, (1, 10))
        field = visualizer.extract_wave_field(input_ids, layer_idx=0)
        
        assert field.shape[1] == base_config.field_size
    
    def test_attention_equivalent(self, base_config):
        """Test attention-equivalent computation."""
        from crumb_llm.model import WaveFieldLM
        from crumb_llm.research.interpretability import AttentionEquivalent
        
        model = WaveFieldLM(base_config)
        attention = AttentionEquivalent(model)
        
        input_ids = torch.randint(0, 256, (1, 10))
        influence = attention.compute_influence_matrix(input_ids, layer_idx=0)
        
        assert influence.shape == (1, 10, 10)
    
    def test_concept_discovery(self, base_config):
        """Test concept discovery."""
        from crumb_llm.model import WaveFieldLM
        from crumb_llm.research.interpretability import ConceptDiscovery
        
        model = WaveFieldLM(base_config)
        concepts = ConceptDiscovery(model)
        
        # Discover a concept
        pos_examples = [torch.randint(0, 256, (10,)) for _ in range(5)]
        neg_examples = [torch.randint(0, 256, (10,)) for _ in range(5)]
        
        direction = concepts.discover_concepts(
            pos_examples, neg_examples, "test_concept"
        )
        
        assert direction.shape == (base_config.dim,)
        assert "test_concept" in concepts.concepts


class TestMetaLearning:
    """Tests for meta-learning."""
    
    def test_maml_optimizer(self, base_config):
        """Test MAML optimizer."""
        from crumb_llm.model import WaveFieldLM
        from crumb_llm.research.meta_learning import MAMLOptimizer, MetaLearningConfig
        
        model = WaveFieldLM(base_config)
        meta_config = MetaLearningConfig(num_inner_steps=2)
        maml = MAMLOptimizer(model, meta_config)
        
        # Test inner loop
        support_ids = torch.randint(0, 256, (5, 10))
        support_targets = torch.randint(0, 256, (5, 10))
        
        adapted_model, loss = maml.inner_loop(support_ids, support_targets)
        assert adapted_model is not None
    
    def test_prototypical_network(self, base_config):
        """Test prototypical network."""
        from crumb_llm.model import WaveFieldLM
        from crumb_llm.research.meta_learning import PrototypicalNetwork
        
        model = WaveFieldLM(base_config)
        proto_net = PrototypicalNetwork(model)
        
        # Compute prototypes
        support_ids = torch.randint(0, 256, (10, 10))  # 2 classes, 5 examples each
        support_labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        
        prototypes = proto_net.compute_prototypes(support_ids, support_labels, 2)
        assert prototypes.shape == (2, base_config.dim)


class TestSparse:
    """Tests for sparse architectures."""
    
    def test_wave_router(self, base_config):
        """Test wave router."""
        from crumb_llm.research.sparse import WaveRouter
        
        router = WaveRouter(base_config.dim, num_experts=4, top_k=2)
        
        x = torch.randn(2, 10, base_config.dim)
        weights, indices, loss = router(x)
        
        assert weights.shape == (2, 10, 2)
        assert indices.shape == (2, 10, 2)
    
    def test_sparse_model(self, base_config):
        """Test sparse wave field model."""
        sparse_config = SparseConfig(num_experts=4, top_k_experts=2)
        model = SparseWaveField(base_config, sparse_config)
        
        input_ids = torch.randint(0, 256, (2, 10))
        out = model(input_ids)
        
        assert "logits" in out
        assert "aux_loss" in out
    
    def test_pruning(self, base_config):
        """Test weight pruning."""
        sparse_config = SparseConfig(sparsity_ratio=0.5)
        model = SparseWaveField(base_config, sparse_config)
        
        # Count parameters before pruning
        params_before = sum(p.numel() for p in model.parameters())
        
        # Prune
        model.prune_weights()
        
        # Count active parameters
        active_params = model.count_active_parameters()
        assert active_params < params_before


class TestLongContext:
    """Tests for long context handling."""
    
    def test_hierarchical_field(self, base_config):
        """Test hierarchical wave field."""
        long_config = LongContextConfig(window_size=64)
        from crumb_llm.research.long_context import HierarchicalField
        
        hierarchical = HierarchicalField(base_config, long_config)
        
        # Test with long sequence
        x = torch.randn(2, 200, base_config.dim)
        out = hierarchical(x)
        
        assert out.shape == x.shape
    
    def test_landmark_attention(self, base_config):
        """Test landmark attention."""
        from crumb_llm.research.long_context import LandmarkAttention
        
        landmark_attn = LandmarkAttention(base_config.dim, num_landmarks=16)
        
        x = torch.randn(2, 100, base_config.dim)
        out = landmark_attn(x)
        
        assert out.shape == x.shape
    
    def test_compressive_memory(self, base_config):
        """Test compressive memory."""
        from crumb_llm.research.long_context import CompressiveMemory
        
        memory = CompressiveMemory(base_config.dim, memory_size=64)
        
        # Store context
        context = torch.randn(2, 50, base_config.dim)
        memory.compress_and_store(context)
        
        # Retrieve
        query = torch.randn(2, 10, base_config.dim)
        retrieved = memory.retrieve(query)
        
        assert retrieved.shape == query.shape
    
    def test_long_context_model(self, base_config):
        """Test complete long context model."""
        long_config = LongContextConfig(window_size=64, num_landmarks=16)
        model = LongContextWaveField(base_config, long_config)
        
        # Test with long sequence
        input_ids = torch.randint(0, 256, (2, 200))
        out = model(input_ids)
        
        assert "logits" in out
        assert out["logits"].shape == (2, 200, 256)


class TestIntegration:
    """Integration tests combining multiple features."""
    
    def test_sparse_long_context(self, base_config):
        """Test combining sparse and long context."""
        # This would test using sparse MoE with long context
        # Implementation would combine both features
        pass
    
    def test_multimodal_meta_learning(self, base_config):
        """Test meta-learning on multi-modal tasks."""
        # This would test few-shot learning on multi-modal data
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Made with Bob
