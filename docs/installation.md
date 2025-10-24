# Installation Guide

This guide will help you set up the Local Hybrid Retrieval-Augmented Document QA system on your local machine.

## System Requirements

### Minimum Requirements

- **Python**: 3.8 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space for models and datasets
- **Operating System**: Windows 10+, macOS 10.15+, or Linux

### Recommended Hardware

- **CPU**: Multi-core processor (4+ cores)
- **RAM**: 16GB or more
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster embeddings)
- **Storage**: SSD with 20GB+ free space

## Prerequisites

### Python Environment

1. **Download and install Python 3.11+** from [python.org](https://python.org)
2. **Verify installation**:
   ```bash
   python --version  # Should be 3.11 or higher
   pip --version
   ```

### Ollama (Local LLM)

1. **Download and install Ollama** from [ollama.com](https://ollama.com)
2. **Pull the Llama 3.2 model**:
   ```bash
   ollama pull llama3.2
   ```
3. **Verify Ollama is running**:
   ```bash
   curl http://localhost:11434/api/tags
   ```

### Required Libraries

The system requires several machine learning and NLP libraries:

- **FAISS**: For efficient vector similarity search
- **PyTorch**: For neural network computations
- **Transformers**: For pre-trained language models
- **Sentence Transformers**: For text embeddings

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd local-hybrid-rag-qa
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Required Models (Optional)

The system will automatically download models on first use, but you can pre-download them:

```bash
# Download sentence transformer model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Download language model (if using local generation)
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-small')"
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Copy the example environment file
cp .env.example .env

# Edit with your settings
nano .env
```

### Configuration File

The main configuration is in `src/config.py`. Key settings to review:

```python
# Retrieval settings
SPARSE_WEIGHT = 0.3
DENSE_WEIGHT = 0.7

# Model settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GENERATION_MODEL = "microsoft/DialoGPT-small"

# Evaluation settings
BOOTSTRAP_SAMPLES = 1000
CONFIDENCE_LEVEL = 0.95
```

## Testing Installation

### 1. Run Basic Tests

```bash
# Test retrieval components
python -c "from src.retrieval.hybrid_retriever import HybridRetriever; print('Retrieval OK')"

# Test generation components
python -c "from src.generation.hallucination_judge import Judge; print('Generation OK')"
```

### 2. Run Evaluation

```bash
python src/evaluation/run_weight_sweep.py
```

### 3. Start Web Interface

```bash
python src/server_MPC.py
```

Navigate to `http://localhost:5000` to verify the web interface works.

## Troubleshooting

### Common Issues

#### FAISS Installation Issues

```bash
# If pip install fails
conda install -c pytorch faiss-cpu

# Or install from source
pip install faiss-cpu
```

#### CUDA/GPU Issues

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA version of PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Memory Issues

```bash
# Reduce batch size in config.py
BATCH_SIZE = 8  # Instead of 32

# Use CPU-only mode
export CUDA_VISIBLE_DEVICES=""
```

#### Model Download Issues

```bash
# Set proxy if needed
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080

# Use local model cache
export TRANSFORMERS_CACHE=/path/to/cache
```

### Performance Optimization

#### For CPU-Only Systems

```python
# In config.py
DEVICE = "cpu"
BATCH_SIZE = 4
MAX_LENGTH = 256
```

#### For GPU Systems

```python
# In config.py
DEVICE = "cuda"
BATCH_SIZE = 32
MAX_LENGTH = 512
```

### Getting Help

If you encounter issues:

1. Check the [GitHub Issues](https://github.com/username/repo/issues) page
2. Review the [Usage Guide](usage.md) for examples
3. Ensure all prerequisites are met
4. Try the troubleshooting steps above

## Next Steps

Once installed, you can:

- [Run the web interface](usage.md#web-interface)
- [Perform evaluations](usage.md#evaluation)
- [Customize the system](usage.md#configuration)
- [Contribute to the project](contributing.md)
