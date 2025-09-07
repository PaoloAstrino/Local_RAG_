# Troubleshooting Guide

This guide helps you resolve common issues when using the Local Hybrid Retrieval-Augmented Document QA system.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Runtime Errors](#runtime-errors)
- [Performance Issues](#performance-issues)
- [Web Interface Issues](#web-interface-issues)
- [Model and Data Issues](#model-and-data-issues)
- [Getting Help](#getting-help)

## Installation Issues

### pip install fails with "No module named 'setuptools'"

**Problem**: setuptools is not installed or outdated.

**Solution**:

```bash
# Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

# Or install setuptools first
pip install setuptools
```

### FAISS installation fails

**Problem**: FAISS compilation issues on some systems.

**Solutions**:

```bash
# Try CPU-only version
pip install faiss-cpu

# Or use conda
conda install -c pytorch faiss-cpu

# For GPU support (if CUDA installed)
pip install faiss-gpu
```

### PyTorch CUDA version mismatch

**Problem**: PyTorch version doesn't match CUDA version.

**Solution**:

```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch version
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Memory error during installation

**Problem**: Not enough RAM for compilation.

**Solution**:

```bash
# Use pre-compiled wheels
pip install --only-binary=all -r requirements.txt

# Or install in smaller batches
pip install torch
pip install transformers
pip install faiss-cpu
```

## Runtime Errors

### "CUDA out of memory"

**Problem**: GPU memory insufficient for model loading.

**Solutions**:

```python
# In config.py or environment
export CUDA_VISIBLE_DEVICES=""
# Forces CPU usage

# Or reduce model size
config.EMBEDDING_MODEL = "all-MiniLM-L3-v2"  # Smaller model
config.BATCH_SIZE = 4
```

### "Model not found" error

**Problem**: Model files not downloaded or corrupted.

**Solution**:

```bash
# Clear cache and re-download
rm -rf ~/.cache/huggingface
rm -rf ~/.cache/torch

# Set cache directory
export TRANSFORMERS_CACHE=/path/to/cache
python -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/DialoGPT-small')"
```

### "UnicodeDecodeError" with special characters

**Problem**: Text encoding issues in documents.

**Solution**:

```python
# Specify encoding when reading files
with open('document.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Or handle encoding errors
with open('document.txt', 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()
```

### "Connection refused" when starting server

**Problem**: Port already in use or firewall blocking.

**Solutions**:

```bash
# Check port usage
netstat -ano | findstr :5000

# Kill process using port
taskkill /PID <PID> /F

# Or use different port
python src/server_MPC.py --port 5001
```

## Performance Issues

### Slow document processing

**Problem**: Large documents or inefficient processing.

**Solutions**:

```python
# Reduce chunk size
config.CHUNK_SIZE = 256
config.CHUNK_OVERLAP = 32

# Use CPU optimization
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4

# Enable GPU if available
export CUDA_VISIBLE_DEVICES=0
```

### High memory usage

**Problem**: Memory leaks or large models.

**Solutions**:

```python
# Clear cache periodically
import gc
gc.collect()

# Use smaller models
config.EMBEDDING_MODEL = "paraphrase-MiniLM-L3-v2"
config.GENERATION_MODEL = "distilgpt2"

# Reduce batch size
config.BATCH_SIZE = 4
```

### Slow query response times

**Problem**: Inefficient retrieval or generation.

**Solutions**:

```python
# Optimize retrieval weights
config.SPARSE_WEIGHT = 0.6  # Favor faster BM25
config.DENSE_WEIGHT = 0.4

# Reduce generation length
config.MAX_NEW_TOKENS = 50

# Use faster models
config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Faster than larger models
```

## Web Interface Issues

### Page doesn't load

**Problem**: Server not running or network issues.

**Solutions**:

```bash
# Check server status
curl http://localhost:5000/status

# Restart server
python src/server_MPC.py

# Check firewall
# Windows: Check Windows Defender Firewall
# Linux: sudo ufw status
```

### File upload fails

**Problem**: File size limits or format issues.

**Solutions**:

```python
# Increase upload limit
config.MAX_UPLOAD_SIZE = 200 * 1024 * 1024  # 200MB

# Check supported formats
# Supported: PDF, TXT, MD, DOCX
```

### JavaScript errors in browser

**Problem**: Browser compatibility or caching issues.

**Solutions**:

```bash
# Clear browser cache
# Chrome: Ctrl+Shift+R (hard refresh)

# Check browser console for errors
# Chrome: F12 → Console tab

# Try different browser
# Recommended: Chrome, Firefox, Edge
```

## Model and Data Issues

### Poor retrieval quality

**Problem**: Low relevance in search results.

**Solutions**:

```python
# Adjust retrieval weights
config.SPARSE_WEIGHT = 0.4
config.DENSE_WEIGHT = 0.6

# Try different embedding model
config.EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-v1"

# Increase top_k
config.TOP_K = 20
```

### Hallucination detection too aggressive

**Problem**: Valid answers flagged as hallucinations.

**Solutions**:

```python
# Adjust threshold
judge = Judge(threshold=0.5)  # Increase from default 0.3

# Use different judge model
judge = Judge(model="api")  # If available
```

### Evaluation metrics inconsistent

**Problem**: Bootstrap results vary between runs.

**Solutions**:

```python
# Set random seed
config.RANDOM_SEED = 42

# Increase bootstrap samples
config.BOOTSTRAP_SAMPLES = 2000

# Check data consistency
# Ensure ground truth is stable
```

## Getting Help

### Debug Information

**Collect system information**:

```bash
# Python version and packages
python --version
pip list | grep -E "(torch|transformers|faiss|sentence-transformers)"

# System information
python -c "import platform; print(platform.platform())"

# GPU information (if applicable)
nvidia-smi
```

**Enable debug logging**:

```bash
export LOG_LEVEL=DEBUG
python src/server_MPC.py
```

### Common Debug Steps

1. **Check Python environment**:

   ```bash
   which python
   python -c "import sys; print(sys.path)"
   ```

2. **Verify model loading**:

   ```python
   from transformers import AutoModel
   model = AutoModel.from_pretrained('microsoft/DialoGPT-small')
   print("Model loaded successfully")
   ```

3. **Test basic functionality**:
   ```python
   from src.retrieval.hybrid_retriever import HybridRetriever
   retriever = HybridRetriever()
   print("Retriever initialized")
   ```

### Where to Get Help

1. **GitHub Issues**: Check existing issues and create new ones
2. **Documentation**: Review all docs in the `docs/` folder
3. **Community**: Join discussions or forums
4. **Logs**: Check application logs for error details

### Creating a Good Bug Report

**Include**:

- Error message and stack trace
- Steps to reproduce
- System information (OS, Python version, hardware)
- Configuration settings
- Sample data (if applicable)

**Example**:

```
OS: Windows 11
Python: 3.9.7
Error: CUDA out of memory

Steps:
1. Start server: python src/server_MPC.py
2. Upload 50MB PDF
3. Ask question
4. Get CUDA error

Config:
BATCH_SIZE = 32
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
```

### Performance Tuning Checklist

- [ ] Use appropriate hardware (GPU for large models)
- [ ] Optimize batch sizes for your system
- [ ] Choose right model sizes for your use case
- [ ] Monitor memory usage during operation
- [ ] Use caching for repeated queries
- [ ] Profile code for bottlenecks

### Emergency Solutions

**If system becomes unresponsive**:

```bash
# Kill all Python processes
taskkill /F /IM python.exe

# Clear caches
rm -rf ~/.cache/huggingface/*
rm -rf __pycache__/
```

**Reset to defaults**:

```python
# In config.py
config = Config()  # Creates default configuration
config.save("config.json")
```

This troubleshooting guide covers the most common issues. If you encounter a problem not listed here, please check the GitHub issues or create a new one with detailed information.
