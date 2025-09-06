# Local Hybrid Retrieval-Augmented Document QA

This repository implements a privacy-preserving, hybrid retrieval-augmented generation (RAG) system for document question-answering (QA). It combines sparse (BM25) and dense (embedding-based) retrieval to optimize performance while minimizing hallucinations.

## Key Features
- Hybrid retrieval with configurable sparse/dense weights
- Hallucination detection using LLM-as-Judge
- Evaluation on SQuAD, MS MARCO, and Natural Questions datasets
- Local deployment for privacy (no external APIs required)

## Quick Start
1. Clone the repo: `git clone <repo-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Run evaluation: `python evaluation/scripts/run_weight_sweep.py`

## Citation
If you use this work, please cite our paper:
```
@article{astrino2025local,
  title={Local Hybrid Retrieval-Augmented Document QA},
  author={Astrino, Paolo},
  journal={arXiv preprint},
  year={2025}
}
```

## License
MIT License. See LICENSE for details.
