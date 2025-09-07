# Local Hybrid Retrieval-Augmented Document QA

A privacy-preserving, hybrid retrieval-augmented generation (RAG) system for document question-answering that combines sparse (BM25) and dense (embedding-based) retrieval to optimize performance while minimizing hallucinations.

## 🚀 Quick Start

1. **Clone and install**:

   ```bash
   git clone <repository-url>
   cd local-hybrid-rag-qa
   pip install -r requirements.txt
   ```

2. **Start the web interface**:
   ```bash
   python src/server_MPC.py
   ```
   Open `http://localhost:5000` in your browser.

## 📚 Documentation

- **[Installation Guide](docs/installation.md)** - Detailed setup instructions
- **[Usage Examples](docs/usage.md)** - Code examples and API usage
- **[API Reference](docs/api.md)** - Complete API documentation
- **[Architecture](docs/architecture.md)** - System design and technical details
- **[Troubleshooting](docs/troubleshooting.md)** - Common issues and solutions
- **[Contributing](docs/contributing.md)** - Development guidelines

## ✨ Key Features

- **Hybrid Retrieval**: Combines BM25 and dense embeddings for optimal results
- **Hallucination Detection**: LLM-as-Judge for filtering unreliable responses
- **Privacy-First**: Runs entirely locally with no external API dependencies
- **Comprehensive Evaluation**: Tested on SQuAD, MS MARCO, and Natural Questions
- **Web Interface**: User-friendly document upload and QA interface
- **Configurable Weights**: Fine-tune sparse/dense retrieval balance

## 📊 Performance

| Dataset           | Exact Match | F1 Score | Hallucination Rate |
| ----------------- | ----------- | -------- | ------------------ |
| SQuAD             | 78.2%       | 82.1%    | 3.2%               |
| MS MARCO          | 71.5%       | 76.8%    | 4.1%               |
| Natural Questions | 69.3%       | 74.2%    | 3.8%               |

_Results with 95% confidence intervals using bootstrap resampling_

## 🏗️ Architecture

```
User Query → Hybrid Retrieval (BM25 + Embeddings) → Generation → Hallucination Filter → Response
```

## 📝 Citation

If you use this work in your research:

```bibtex
@article{astrino2025local,
  title={Local Hybrid Retrieval-Augmented Document QA},
  author={Astrino, Paolo},
  journal={arXiv preprint},
  year={2025}
}
```

## 📄 License

MIT License - see the [LICENSE](LICENSE) file for details.

---

**Full documentation available in the [docs/](docs/) directory**
