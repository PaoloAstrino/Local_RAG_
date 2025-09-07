# Contributing Guide

We welcome contributions to the Local Hybrid Retrieval-Augmented Document QA project! This guide will help you get started with development, testing, and contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style and Standards](#code-style-and-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)
- [Documentation](#documentation)

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)
- Code editor (VS Code, PyCharm, etc.)

### Fork and Clone

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/your-username/local-hybrid-rag-qa.git
   cd local-hybrid-rag-qa
   ```
3. **Set up upstream remote**:
   ```bash
   git remote add upstream https://github.com/original-owner/local-hybrid-rag-qa.git
   ```

## Development Setup

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### 2. Pre-commit Hooks

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run hooks on all files
pre-commit run --all-files
```

### 3. Development Configuration

```bash
# Copy development config
cp config/dev.config.json config/local.config.json

# Edit for your environment
nano config/local.config.json
```

## Code Style and Standards

### Python Style Guide

We follow PEP 8 with some modifications:

```python
# Good: Clear, readable code
def process_documents(documents: List[Document],
                     config: ProcessingConfig) -> List[ProcessedDocument]:
    """Process a list of documents according to configuration."""
    processed = []
    for doc in documents:
        if doc.is_valid():
            processed_doc = _process_single_document(doc, config)
            processed.append(processed_doc)
    return processed

# Bad: Unclear, violates conventions
def proc_docs(docs,conf):
    procd=[]
    for d in docs:
        if d.is_valid:
            pd=_proc_sing_doc(d,conf)
            procd.append(pd)
    return procd
```

### Key Standards

#### Naming Conventions

- **Functions/Methods**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private members**: `_leading_underscore`

#### Documentation

- **Docstrings**: Use Google-style docstrings
- **Type Hints**: Required for all functions
- **Comments**: Explain why, not what

```python
def retrieve_documents(query: str,
                      corpus: List[str],
                      top_k: int = 10) -> List[DocumentResult]:
    """
    Retrieve top-k relevant documents for a query.

    Args:
        query: The search query string
        corpus: List of document texts to search
        top_k: Number of top results to return

    Returns:
        List of document results with scores

    Raises:
        ValueError: If query is empty or corpus is empty
    """
    if not query.strip():
        raise ValueError("Query cannot be empty")
    if not corpus:
        raise ValueError("Corpus cannot be empty")

    # Implementation here
    pass
```

#### Error Handling

```python
# Good: Specific exceptions with context
def load_model(model_path: str) -> Model:
    try:
        model = torch.load(model_path)
        model.eval()
        return model
    except FileNotFoundError:
        raise ModelLoadError(f"Model file not found: {model_path}")
    except Exception as e:
        raise ModelLoadError(f"Failed to load model: {str(e)}")

# Bad: Generic exception handling
def load_model(model_path: str) -> Model:
    try:
        return torch.load(model_path)
    except:
        raise Exception("Model loading failed")
```

### Code Organization

#### File Structure

```
src/
├── retrieval/           # Retrieval components
│   ├── __init__.py
│   ├── hybrid_retriever.py
│   ├── sparse_retriever.py
│   └── dense_retriever.py
├── generation/          # Generation components
│   ├── __init__.py
│   ├── generator.py
│   └── hallucination_judge.py
├── evaluation/          # Evaluation framework
│   ├── __init__.py
│   ├── evaluator.py
│   └── bootstrap.py
└── utils/              # Utilities
    ├── __init__.py
    ├── config.py
    └── logging.py
```

#### Import Organization

```python
# Standard library imports
import os
import sys
from typing import List, Dict, Optional

# Third-party imports
import torch
import numpy as np
from transformers import AutoModel

# Local imports
from .config import Config
from ..utils.logging import get_logger
```

## Testing

### Test Structure

```
tests/
├── unit/                # Unit tests
│   ├── test_retrieval.py
│   ├── test_generation.py
│   └── test_evaluation.py
├── integration/         # Integration tests
│   ├── test_full_pipeline.py
│   └── test_web_interface.py
└── fixtures/           # Test data
    ├── sample_documents.json
    └── test_queries.json
```

### Writing Tests

```python
import pytest
from unittest.mock import Mock, patch

class TestHybridRetriever:
    @pytest.fixture
    def sample_corpus(self):
        return [
            "Machine learning is a subset of AI.",
            "Deep learning uses neural networks.",
            "Natural language processing handles text."
        ]

    @pytest.fixture
    def retriever(self):
        return HybridRetriever(sparse_weight=0.5, dense_weight=0.5)

    def test_retrieve_returns_correct_number_of_results(self, retriever, sample_corpus):
        query = "What is machine learning?"
        results = retriever.retrieve(query, sample_corpus, top_k=2)

        assert len(results) == 2
        assert all('text' in result for result in results)
        assert all('score' in result for result in results)

    @patch('src.retrieval.sparse_retriever.BM25Retriever')
    def test_retrieve_handles_empty_corpus(self, mock_sparse, retriever):
        mock_sparse.return_value.score.return_value = []

        with pytest.raises(ValueError, match="Corpus cannot be empty"):
            retriever.retrieve("test query", [])
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_retrieval.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/unit/test_retrieval.py::TestHybridRetriever::test_retrieve_returns_correct_number_of_results
```

### Test Coverage Requirements

- **Minimum Coverage**: 80%
- **Critical Paths**: 90%+ coverage
- **New Features**: 100% coverage required

## Submitting Changes

### Development Workflow

1. **Create a feature branch**:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding standards

3. **Write tests** for new functionality

4. **Run the test suite**:

   ```bash
   pytest
   pre-commit run --all-files
   ```

5. **Update documentation** if needed

6. **Commit your changes**:

   ```bash
   git add .
   git commit -m "feat: add new feature

   - Add detailed description
   - Reference issue numbers
   - Explain breaking changes"
   ```

7. **Push to your fork**:

   ```bash
   git push origin feature/your-feature-name
   ```

8. **Create a Pull Request** on GitHub

### Commit Message Format

```
type(scope): description

[optional body]

[optional footer]
```

**Types**:

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Testing
- `chore`: Maintenance

**Examples**:

```
feat(retrieval): add support for custom embedding models

- Add EmbeddingModelConfig class
- Support multiple sentence transformers
- Add model validation

Closes #123
```

```
fix(evaluation): handle empty result sets in bootstrap

- Add check for empty predictions
- Return default confidence intervals
- Add test case for edge case
```

### Pull Request Guidelines

- **Title**: Clear, descriptive title
- **Description**: Detailed explanation of changes
- **Tests**: Include test results and coverage
- **Breaking Changes**: Clearly marked and documented
- **Screenshots**: For UI changes
- **Checklist**: Complete the PR template

## Reporting Issues

### Bug Reports

**Good Bug Report**:

- Clear title describing the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Error messages and stack traces
- Minimal code example if applicable

**Example**:

```
Title: Retrieval fails with special characters in query

Description:
When querying with special characters like "café" or "naïve", the retrieval system fails with a UnicodeDecodeError.

Steps to reproduce:
1. Start the server with `python src/server_MPC.py`
2. Upload a document containing "café"
3. Query for "café information"
4. See error in logs

Expected: Successful retrieval
Actual: UnicodeDecodeError in sparse_retriever.py:45

Environment:
- OS: Windows 11
- Python: 3.9.7
- Browser: Chrome 91.0
```

### Feature Requests

**Good Feature Request**:

- Clear description of the proposed feature
- Use case and benefits
- Implementation suggestions (optional)
- Mockups or examples (if applicable)

## Documentation

### Documentation Standards

- **README**: Keep updated with latest features
- **Code Comments**: Explain complex logic
- **API Docs**: Document all public functions
- **Examples**: Provide working code examples

### Updating Documentation

```bash
# Update API documentation
sphinx-apidoc -f -o docs/api src/

# Build documentation
cd docs
make html

# Serve locally
python -m http.server 8000
```

## Code Review Process

### Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] No breaking changes without discussion
- [ ] Performance impact considered
- [ ] Security implications reviewed

### Review Comments

- **Be constructive**: Focus on code improvement
- **Explain reasoning**: Why a change is needed
- **Suggest alternatives**: When rejecting an approach
- **Acknowledge good work**: Positive feedback is important

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers learn
- Maintain professional communication

### Getting Help

- **Documentation**: Check docs first
- **Issues**: Search existing issues
- **Discussions**: Use GitHub Discussions for questions
- **Slack/Discord**: Join community chat (if available)

## Recognition

Contributors will be:

- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Acknowledged in the paper (for significant contributions)

Thank you for contributing to the Local Hybrid Retrieval-Augmented Document QA project!
