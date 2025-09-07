# Local Hybrid Retrieval-Augmented Document QA

This repository implements a privacy-preserving, hybrid retrieval-augmented generation (RAG) system for document question-answering (QA). It combines sparse (BM25) and dense (embedding-based) retrieval to optimize performance while minimizing hallucinations.

## Key Features

- **Hybrid Retrieval**: Combines BM25 (sparse) and dense embeddings for optimal retrieval
- **Hallucination Detection**: Uses LLM-as-Judge for filtering unreliable responses
- **Privacy-Preserving**: Runs entirely locally with no external API dependencies
- **Comprehensive Evaluation**: Tested on SQuAD, MS MARCO, and Natural Questions datasets
- **Configurable Weights**: Fine-tune sparse/dense retrieval balance
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

4. **Start the web interface**
   ```bash
   python src/server_MPC.py
   ```
   Then open `http://localhost:5000` in your browser.

## Project Structure

```
├── src/                    # Source code
│   ├── client_MPC.py      # Client application
│   ├── server_MPC.py      # Server application
│   ├── config.py          # Configuration settings
│   ├── retrieval/         # Retrieval components
│   ├── generation/        # Generation components
│   └── utils/             # Utility functions
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
