# Installation Guide

## Prerequisites
- Python 3.8+
- FAISS (for vector search)
- PyTorch (for embeddings)

## Steps
1. Create a virtual environment: `python -m venv venv`
2. Activate: `venv\Scripts\activate` (Windows)
3. Install: `pip install -r requirements.txt`
4. Download datasets (e.g., SQuAD) and place in `evaluation/results/`

## Troubleshooting
- If FAISS fails, install via conda: `conda install -c pytorch faiss-cpu`
