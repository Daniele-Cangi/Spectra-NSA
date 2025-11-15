# Contributing to NSA (Neural Semantic Architecture)

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Avoid discriminatory or harassing language
- Focus on constructive feedback
- Help others learn and grow

## Before You Start

1. **Fork** the repository
2. **Create a branch** for your feature (`git checkout -b feature/amazing-feature`)
3. **Read** the [README.md](../README.md) and [ULTIMATE_GUIDE.md](../ULTIMATE_GUIDE.md)

## Areas for Contribution

### Priority Areas ⭐
- Multi-GPU/distributed training optimization
- Mixed precision (FP16) stability improvements
- Gradient checkpointing implementation
- Memory optimization techniques
- Additional evaluation benchmarks
- Documentation improvements

### Bug Reports
- Clearly describe the bug
- Provide steps to reproduce
- Include environment info (GPU, CUDA version, PyTorch version)
- Attach error logs

### Feature Requests
- Describe the use case
- Explain why it's needed
- Provide implementation ideas if possible

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/NSA_2.0.git
cd NSA_2.0

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks (optional)
pre-commit install
```

## Code Standards

### Style
- **Python**: Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- **Line length**: 100 characters
- **Naming**: 
  - Classes: `PascalCase`
  - Functions/variables: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`

### Type Hints
Use type hints for all function signatures:
```python
def compute_loss(embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute contrastive loss."""
    ...
```

### Documentation
- Add docstrings to all classes and functions
- Use Google-style docstrings
- Include examples for complex functions

```python
def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Forward pass through model.
    
    Args:
        x: Input dict with keys: input_ids, attention_mask
        
    Returns:
        Dict with keys: embedding_semantic, embedding_ranking, spectral, anomaly_score
        
    Example:
        >>> model = FullEmbedder(cfg)
        >>> input = {"input_ids": ..., "attention_mask": ...}
        >>> output = model(input)
    """
```

### Comments
- Use comments to explain **why**, not what
- Keep comments concise and clear
- Update comments when code changes

## Testing

```bash
# Run tests
pytest tests/

# Run specific test
pytest tests/test_model.py::test_forward_pass

# Run with coverage
pytest --cov=. tests/
```

### Writing Tests
```python
# tests/test_model.py
import pytest
import torch
from anomalous_embedding_ultimate import FullEmbedder, Config

def test_forward_pass():
    """Test model forward pass."""
    cfg = Config(size="M456")
    model = FullEmbedder(cfg)
    
    batch_size = 2
    seq_len = 256
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    output = model({
        "input_ids": input_ids,
        "attention_mask": attention_mask
    })
    
    assert "embedding_semantic" in output
    assert output["embedding_semantic"][768].shape == (batch_size, 768)
```

## Pull Request Process

1. **Update** the [CHANGELOG.md](../CHANGELOG.md) with your changes
2. **Ensure** all tests pass: `pytest tests/`
3. **Run** linters: `black . && flake8 . && mypy .`
4. **Write** clear PR title and description
5. **Link** related issues: "Fixes #123"
6. **Request** review from maintainers

### PR Template
```markdown
## Description
Brief description of changes

## Type
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Optimization

## Changes
- Point 1
- Point 2

## Testing
Describe tests added/modified

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added
- [ ] Tests pass
- [ ] No new warnings
```

## Commit Guidelines

Use conventional commits:
```bash
git commit -m "feat: add gradient checkpointing"
git commit -m "fix: resolve NaN in spectral loss"
git commit -m "docs: update README with examples"
git commit -m "refactor: simplify attention mechanism"
git commit -m "test: add model forward pass tests"
```

## Documentation

- Keep [README.md](../README.md) updated
- Update [ULTIMATE_GUIDE.md](../ULTIMATE_GUIDE.md) for new features
- Add docstrings to new code
- Create examples if applicable

## Review Process

- At least 1 maintainer review required
- All CI checks must pass
- Code coverage should not decrease
- Performance benchmarks may be requested for optimization PRs

## Questions?

- Open an issue with the label `question`
- Check existing issues and discussions
- Email: contributors@example.com

---

## Acknowledgments

Contributors are acknowledged in:
1. Commit history
2. GitHub contributors page
3. [CONTRIBUTORS.md](../CONTRIBUTORS.md) (if applicable)

Thank you for making NSA better! 🙏
