# Local Hybrid Retrieval-Augmented Document QA

This repository implements a privacy-preserving, hybrid retrieval-augmented generation (RAG) system for document question-answering (QA). It combines sparse (BM25) and dense (embedding-based) retrieval to optimize performance while minimizing hallucinations.

## Key Features

- **100% Local Operation**: All processing (embeddings, retrieval, LLM inference) runs on-premises via Ollama/Llama 3.2
- **Hybrid Retrieval**: Combines BM25 (sparse) and BGE embeddings (dense) for optimal retrieval
- **Zero External Dependencies**: Complete data sovereignty with no cloud API calls
- **Hallucination Detection**: LLM-as-Judge evaluation for reliability assessment
- **Comprehensive Evaluation**: Tested on SQuAD, MS MARCO, and Natural Questions datasets
- **GPU Acceleration**: 4.2× speedup for embeddings, 3× for LLM inference
- **Configurable Weights**: Fine-tune sparse/dense retrieval balance (optimal: 30/70)
- **Bootstrap Evaluation**: Statistical confidence intervals for robust results

## Quick Start

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd local-hybrid-rag-qa
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run evaluation**

   ```bash
   python src/evaluation/run_weight_sweep.py
   ```

4. **Start the system**

   ```bash
   # Terminal 1: Start the server
   python src/server.py

   # Terminal 2: Start the client
   python src/client.py
   ```

   Then open `http://localhost:5001` in your browser.

## Project Structure

```
├── src/                    # Source code
│   ├── client.py          # Flask HTTP API client
│   ├── server.py          # Core RAG server (retrieval + LLM)
│   └── config.py          # Configuration settings
├── frontend/              # Web interface
│   ├── chat_MPC.html     # Main HTML interface
│   └── static/           # CSS and JavaScript files
├── evaluation/           # Evaluation framework
│   ├── results/          # Evaluation results
│   └── scripts/          # Evaluation scripts
├── docs/                 # Documentation
├── papers/               # Research paper and figures
└── requirements.txt      # Python dependencies
```

## Documentation

- [Installation Guide](installation.md) - Detailed setup instructions
- [Usage Examples](usage.md) - Code examples and API usage
- [API Reference](api.md) - Complete API documentation
- [Architecture](architecture.md) - System design and components
- [Troubleshooting](troubleshooting.md) - Common issues and solutions
- [Contributing](contributing.md) - Development guidelines

## Research Paper

The full research paper is available in the `papers/` directory:

- [RAG_paoloastrino.tex](papers/RAG_paoloastrino.tex) - LaTeX source
- `papers/images/` - Figures and diagrams

## Citation

If you use this work in your research, please cite:

```bibtex
@article{astrino2025local,
  title={Local Hybrid Retrieval-Augmented Document QA},
  author={Astrino, Paolo},
  journal={arXiv preprint},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## Contact

For questions or issues, please open a GitHub issue or contact the maintainer.
