# Local Hybrid Retrieval-Augmented Document QA

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A **100% local**, privacy-preserving retrieval-augmented generation (RAG) system for document question-answering. Combines sparse (BM25) and dense (BGE embeddings) retrieval with local LLM inference (Ollama/Llama 3.2) to optimize performance while maintaining complete data sovereignty.

> **Associated Paper**: "Local Hybrid Retrieval-Augmented Document QA" by Paolo Astrino  
> **Università Ca' Foscari Venezia** | Master's Thesis 2025

## 🎥 Demo

Watch a live demonstration of the chatbot answering questions about a CV:

<video src="demo_chatbot.mp4" controls width="640" height="360">
  Your browser does not support the video tag. [Download the demo video](demo_chatbot.mp4)
</video>

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- CUDA-capable GPU (optional, for acceleration)

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/PaoloAstrino/Local_RAG.git
   cd Local_RAG
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Pull the Llama model**:

   ```bash
   ollama pull llama3.2
   ```

4. **Start the system**:

   ```bash
   # Terminal 1: Start the server
   python src/server.py

   # Terminal 2: Start the client
   python src/client.py
   ```

5. **Open the web interface**:
   ```
   http://localhost:5001
   ```

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

If you use this work in your research, please cite:

**BibTeX:**
```bibtex
@article{astrino2025local,
  title={Local Hybrid Retrieval-Augmented Document QA},
  author={Astrino, Paolo},
  journal={arXiv preprint arXiv:2511.10297},
  year={2025},
  eprint={2511.10297},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2511.10297},
  doi={10.48550/arXiv.2511.10297}
}
```

**Text:**
Astrino, P. (2025). Local Hybrid Retrieval-Augmented Document QA. arXiv preprint arXiv:2511.10297 [cs.CL].

**arXiv Details:**
- **arXiv ID:** 2511.10297
- **Category:** Computation and Language (cs.CL)
- **ACM Classes:** I.2.7; H.3.3
- **Pages:** 10 pages, 5 figures, 3 tables
- **Format:** Conference-style (ACL format)
- **DOI:** https://doi.org/10.48550/arXiv.2511.10297
- **Submitted:** November 13, 2025

## 📄 License

MIT License - see the [LICENSE](LICENSE) file for details.

---

**Full documentation available in the [docs/](docs/) directory**
