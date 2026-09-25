# Contributing to CRUMB LLM

Thank you for your interest in contributing to CRUMB LLM! This document provides guidelines and instructions for contributing.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Making Changes](#making-changes)
5. [Testing](#testing)
6. [Code Style](#code-style)
7. [Pull Request Process](#pull-request-process)
8. [Areas for Contribution](#areas-for-contribution)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of experience level, background, or identity.

### Expected Behavior

- Be respectful and considerate
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Assume good intentions
- Respect differing viewpoints

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Personal attacks or trolling
- Publishing others' private information
- Any conduct that would be inappropriate in a professional setting

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- Git
- Familiarity with language models and deep learning

### First Contributions

Good first issues are labeled with `good-first-issue` on GitHub. These are typically:
- Documentation improvements
- Bug fixes
- Small feature additions
- Test coverage improvements

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/crumb-format.git
cd crumb-format
```

### 2. Create Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[llm,dev]"
```

### 3. Install Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

### 4. Verify Setup

```bash
# Run tests
pytest tests/

# Check imports
python -c "import crumb_llm; print('OK')"
```

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feature/add-adaptive-kernels` - New features
- `fix/nan-loss-issue` - Bug fixes
- `docs/improve-api-reference` - Documentation
- `test/add-cache-tests` - Tests
- `refactor/simplify-scatter` - Code refactoring

### Commit Messages

Follow conventional commits format:

```
type(scope): brief description

Longer description if needed.

Fixes #123
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `style`: Code style changes
- `chore`: Maintenance tasks

**Examples:**
```
feat(kernels): add adaptive kernel support

Implements input-conditioned wave parameters that adapt
based on sequence content.

Fixes #45
```

```
fix(training): prevent NaN loss with gradient clipping

Added gradient clipping to prevent exploding gradients
during training.

Fixes #67
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_crumb_llm_kernels.py

# Run with coverage
pytest tests/ --cov=crumb_llm --cov-report=html

# Run specific test
pytest tests/test_crumb_llm_kernels.py::test_wave_kernel_time -v
```

### Writing Tests

All new code should include tests:

```python
import pytest
import torch
from crumb_llm import wave_kernel_time

def test_wave_kernel_properties():
    """Test wave kernel mathematical properties."""
    kernel = wave_kernel_time(alpha=0.1, omega=2.0, phi=0.0, size=1024)
    
    # Test shape
    assert kernel.shape == (1024,)
    
    # Test symmetry
    assert torch.allclose(kernel, kernel.flip(0), atol=1e-6)
    
    # Test normalization
    assert kernel.abs().max() <= 1.0

def test_wave_kernel_damping():
    """Test that damping reduces amplitude."""
    kernel_low = wave_kernel_time(alpha=0.1, omega=2.0, phi=0.0, size=1024)
    kernel_high = wave_kernel_time(alpha=1.0, omega=2.0, phi=0.0, size=1024)
    
    # Higher damping should have lower amplitude at edges
    assert kernel_low[0] > kernel_high[0]
```

### Test Coverage

- Aim for >90% code coverage
- Test edge cases and error conditions
- Include integration tests for major features
- Add regression tests for bug fixes

## Code Style

### Python Style

Follow PEP 8 with these specifics:

```python
# Good
def wave_kernel_time(
    alpha: float,
    omega: float,
    phi: float,
    size: int,
) -> torch.Tensor:
    """Build wave kernel in time domain.
    
    Args:
        alpha: Damping coefficient (α > 0)
        omega: Angular frequency
        phi: Phase shift
        size: Kernel size
        
    Returns:
        Wave kernel of shape [size]
    """
    t = torch.arange(size) - size // 2
    kernel = torch.exp(-alpha * t.abs()) * torch.cos(omega * t + phi)
    return kernel
```

### Type Hints

Use type hints for all public APIs:

```python
from typing import Optional, Tuple, Dict, Any
import torch
from torch import Tensor

def forward(
    self,
    x: Tensor,
    targets: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    """Forward pass."""
    ...
```

### Documentation

All public functions need docstrings:

```python
def scatter_linear(
    tokens: Tensor,
    positions: Tensor,
    field_size: int,
    weights: Optional[Tensor] = None,
) -> Tensor:
    """Scatter tokens onto continuous field using linear interpolation.
    
    Maps discrete token embeddings to a continuous 1D field by distributing
    each token's state to nearby field positions using linear interpolation.
    
    Args:
        tokens: Token embeddings [B, N, D]
        positions: Token positions [B, N] (0 to N-1)
        field_size: Size of output field
        weights: Optional scatter weights [B, N]
        
    Returns:
        Field state [B, field_size, D]
        
    Example:
        >>> tokens = torch.randn(4, 128, 512)
        >>> positions = torch.arange(128).expand(4, -1)
        >>> field = scatter_linear(tokens, positions, field_size=256)
        >>> field.shape
        torch.Size([4, 256, 512])
    """
    ...
```

### Code Organization

```python
# Imports: standard library, third-party, local
import math
from typing import Optional

import torch
import torch.nn as nn

from crumb_llm.kernels import wave_kernel_freq

# Constants
DEFAULT_FIELD_SIZE = 1024
MAX_SEQUENCE_LENGTH = 32768

# Classes and functions
class WaveFieldBlock(nn.Module):
    """Wave field transformer block."""
    ...
```

## Pull Request Process

### 1. Before Submitting

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages follow conventions
- [ ] Branch is up to date with main

### 2. Create Pull Request

**Title:** Clear, descriptive summary

**Description template:**
```markdown
## Description
Brief description of changes.

## Motivation
Why is this change needed?

## Changes
- List of specific changes
- Another change

## Testing
How was this tested?

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Code follows style guide
- [ ] Backward compatible (or breaking change documented)

Fixes #issue_number
```

### 3. Review Process

- Maintainers will review within 1-2 weeks
- Address feedback promptly
- Keep discussions focused and professional
- Be open to suggestions

### 4. After Approval

- Maintainer will merge your PR
- Your contribution will be in the next release
- You'll be added to contributors list

## Areas for Contribution

### High Priority

1. **Performance Optimizations**
   - Faster FFT implementations
   - Better GPU utilization
   - Memory optimizations

2. **New Features**
   - Adaptive kernels
   - Hybrid wave-attention
   - Multi-dimensional fields

3. **Documentation**
   - More examples
   - Tutorial notebooks
   - Video tutorials

4. **Testing**
   - Increase coverage
   - Benchmark suite expansion
   - Integration tests

### Medium Priority

1. **Model Architectures**
   - Larger model configs
   - Specialized architectures
   - Domain-specific models

2. **Training Improvements**
   - Better data loading
   - Advanced optimizers
   - Curriculum learning

3. **Inference Optimization**
   - Better quantization
   - Faster generation
   - Batch optimization

### Research Contributions

1. **Novel Architectures**
   - New wave propagation methods
   - Alternative scatter/gather strategies
   - Hybrid approaches

2. **Theoretical Analysis**
   - Mathematical proofs
   - Complexity analysis
   - Convergence guarantees

3. **Benchmarking**
   - New benchmark tasks
   - Comparison studies
   - Ablation studies

## Getting Help

### Questions

- **GitHub Discussions**: For questions and ideas
- **GitHub Issues**: For bugs and feature requests
- **Documentation**: Check docs/ first

### Mentorship

New contributors can request mentorship:
1. Comment on a `good-first-issue`
2. Tag `@maintainers` for guidance
3. Join community discussions

## Recognition

Contributors are recognized in:
- README.md contributors section
- Release notes
- Academic papers (for significant contributions)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## Quick Reference

### Common Commands

```bash
# Setup
git clone https://github.com/YOUR_USERNAME/crumb-format.git
cd crumb-format
pip install -e ".[llm,dev]"

# Development
git checkout -b feature/my-feature
# Make changes
pytest tests/
git add .
git commit -m "feat: add my feature"
git push origin feature/my-feature

# Testing
pytest tests/                          # All tests
pytest tests/test_file.py             # Specific file
pytest tests/ --cov=crumb_llm         # With coverage
pytest -v -s                          # Verbose with output

# Code quality
black crumb_llm/                      # Format code
flake8 crumb_llm/                     # Lint
mypy crumb_llm/                       # Type check
```

### Resources

- [Architecture Guide](docs/ARCHITECTURE.md)
- [API Reference](docs/API_REFERENCE.md)
- [Examples](docs/EXAMPLES.md)
- [FAQ](docs/FAQ.md)

---

**Thank you for contributing to CRUMB LLM!** 🌊

Your contributions help make language modeling more efficient and accessible for everyone.