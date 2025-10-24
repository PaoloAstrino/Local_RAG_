# Reproducibility Guide

This document provides detailed instructions for reproducing the results presented in the paper "Local Hybrid Retrieval-Augmented Document QA" by Paolo Astrino.

## Hardware Requirements

- **CPU**: 12-core AMD Ryzen (3.8 GHz boost) or equivalent
- **GPU**: NVIDIA RTX 4050 (6GB VRAM) or better (optional but recommended)
- **RAM**: 32 GB DDR5 (minimum 16 GB)
- **Storage**: NVMe SSD with 50 GB free space

## Software Environment

- **OS**: Windows 11, Linux, or macOS
- **Python**: 3.11.x
- **CUDA**: 11.8+ (if using GPU)
- **Ollama**: Latest version

## Setup Instructions

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/PaoloAstrino/Local_RAG.git
cd Local_RAG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Models

```bash
# Pull Llama 3.2 via Ollama
ollama pull llama3.2

# BGE embeddings will download automatically on first use
```

### 3. Run Evaluation

```bash
# Hybrid retrieval weight sweep (SQuAD)
python evaluation/run_squad_benchmark.py

# MS MARCO evaluation
python evaluation/run_marco_benchmark.py

# Natural Questions evaluation
python evaluation/run_nq_benchmark.py

# Hallucination detection
python evaluation/run_hallucination_eval.py --dataset squad --n-samples 500
```

## Expected Results

### Retrieval Performance (30% sparse / 70% dense)

| Dataset           | MRR   | Recall@10 | Answer Coverage |
| ----------------- | ----- | --------- | --------------- |
| SQuAD             | 0.805 | 0.974     | 0.980           |
| MS MARCO          | 0.250 | 0.620     | 0.487           |
| Natural Questions | 0.813 | 0.978     | 0.987           |

### Hallucination Rates

| Dataset  | Rate | Faithfulness | Confidence |
| -------- | ---- | ------------ | ---------- |
| SQuAD    | 0.8% | 4.93         | 4.87       |
| MS MARCO | 6.2% | 4.79         | 4.71       |

## Variance and Confidence Intervals

Results include 95% confidence intervals computed via 1,000 bootstrap resamples. Minor variations (±0.01 MRR, ±0.5% Recall) are expected due to:

- GPU non-determinism (cuBLAS operations)
- Random sampling in stratified evaluation
- LLM-as-Judge temperature (0.6)

For stricter reproducibility, set:

```bash
export CUBLAS_WORKSPACE_CONFIG=:16:8
```

## Troubleshooting

### GPU Not Detected

```python
import torch
print(torch.cuda.is_available())  # Should return True
```

### Ollama Connection Issues

```bash
# Verify Ollama is running
curl http://localhost:11434/api/tags

# Restart if needed
ollama serve
```

### Memory Errors

Reduce batch size in `src/config.py`:

```python
EMBEDDING_BATCH_SIZE = 16  # Default: 32
```

## Data Artifacts

All evaluation results are provided in `evaluation/results/`:

- Raw CSVs with per-query metrics
- Aggregated analysis (`comprehensive_results_analysis.md`)
- Visualizations (`visualizations/` folder)

## Citation

If you reproduce these results, please cite:

```bibtex
@mastersthesis{astrino2025localrag,
  author = {Astrino, Paolo},
  title = {Local Hybrid Retrieval-Augmented Document QA},
  school = {Università Ca' Foscari Venezia},
  year = {2025},
  url = {https://github.com/PaoloAstrino/Local_RAG}
}
```

## Contact

For questions or issues with reproduction:

- **Email**: paoloastrino01@gmail.com
- **GitHub Issues**: https://github.com/PaoloAstrino/Local_RAG/issues
