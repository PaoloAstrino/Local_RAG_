# Usage Guide

This guide provides examples and instructions for using the Local Hybrid Retrieval-Augmented Document QA system.

## Table of Contents

- [Quick Start](#quick-start)
- [Web Interface](#web-interface)
- [Python API](#python-api)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Advanced Usage](#advanced-usage)

## Quick Start

### Web Interface

1. **Ensure Ollama is running**:

   ```bash
   ollama serve  # If not already running
   ```

2. **Start the server** (Terminal 1):

   ```bash
   python src/server.py
   ```

3. **Start the client** (Terminal 2):

   ```bash
   python src/client.py
   ```

4. **Open your browser** to `http://localhost:5001`

5. **Upload documents** and ask questions through the web interface

### Command Line

```bash
# The system is designed for web interface use
# Direct Python API usage requires importing from src/server.py
# See the web interface at http://localhost:5001 for full functionality
```

## Web Interface

### Features

- **Document Upload**: Support for PDF, TXT, and other text formats
- **Real-time QA**: Ask questions about uploaded documents
- **Confidence Scores**: View retrieval and generation confidence
- **Hallucination Detection**: Automatic filtering of unreliable responses

### Usage Steps

1. **Upload Documents**: Click "Choose Files" and select your documents
2. **Ask Questions**: Type your question in the input field
3. **View Results**: See the answer with confidence scores and source references
4. **Adjust Settings**: Modify retrieval weights and other parameters

## Python API

### Using the Server Directly

The system is designed as a client-server architecture. For programmatic access, you can interact with the server via socket commands:

```python
import socket
import json

def send_command(command_dict):
    """Send command to RAG server"""
    HOST = '127.0.0.1'
    PORT = 65432

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        data = json.dumps(command_dict).encode('utf-8') + b'\n'
        s.sendall(data)

        response = b''
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            response += chunk
            if response.endswith(b'\n'):
                break

        return json.loads(response.decode('utf-8'))

# Upload files
result = send_command({
    "command": "upload",
    "file_paths": ["uploads/document.pdf"]
})

# Ask question
result = send_command({
    "command": "ask",
    "question": "What is the main topic of this document?"
})
print(result['answer'])
```

### Configuration

The hybrid retriever weights can be adjusted in `src/server.py`:

```python
# In the load_db function, modify EnsembleRetriever weights:
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, semantic_retriever],
    weights=[0.3, 0.7]  # [sparse_weight, dense_weight]
)
```

Optimal configuration (from paper): **30% sparse, 70% dense**

## Configuration

### Key Parameters

#### Retrieval Settings (in `src/server.py`)

```python
# Hybrid retriever weights (optimal from paper)
SPARSE_WEIGHT = 0.3          # Weight for BM25 (30%)
DENSE_WEIGHT = 0.7           # Weight for dense embeddings (70%)
TOP_K = 5                    # Number of documents to retrieve

# Embedding model
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"  # HuggingFace embeddings

# Document chunking
CHUNK_SIZE = 400             # Characters per chunk
CHUNK_OVERLAP = 150          # Overlap between chunks
```

#### Generation Settings (Ollama)

```python
# Local LLM via Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2"           # Ollama model name

# Generation parameters
TEMPERATURE = 0.6            # Response randomness (0.0 to 2.0)
TOP_P = 0.95                 # Nucleus sampling
TOP_K = 40                   # Top-k sampling
MAX_TOKENS = 1024            # Maximum response length
```

#### Evaluation Settings (in `src/config.py`)

```python
BOOTSTRAP_SAMPLES = 1000     # Bootstrap resampling iterations
CONFIDENCE_LEVEL = 0.95      # Confidence interval level
RANDOM_SEED = 42             # Random seed for reproducibility
```

### Environment Variables

```bash
# Model cache directory
export TRANSFORMERS_CACHE=/path/to/cache

# CUDA settings
export CUDA_VISIBLE_DEVICES=0

# Logging level
export LOG_LEVEL=INFO
```

## Evaluation

### Running Evaluations

#### Single Dataset

```bash
python src/evaluation/evaluate_dataset.py \
    --dataset squad \
    --sparse_weight 0.3 \
    --dense_weight 0.7 \
    --output results/squad_results.json
```

#### Multiple Datasets

```bash
python src/evaluation/run_weight_sweep.py \
    --datasets squad marco natural_questions \
    --weight_range 0.0 1.0 0.1 \
    --bootstrap_samples 1000
```

#### Custom Evaluation

```python
from src.evaluation.evaluator import Evaluator

evaluator = Evaluator()
results = evaluator.evaluate(
    predictions=your_predictions,
    ground_truth=your_ground_truth,
    metrics=['exact_match', 'f1', 'rouge']
)

print(f"Exact Match: {results['exact_match']:.3f}")
print(f"F1 Score: {results['f1']:.3f}")
```

### Evaluation Metrics

- **Exact Match**: Percentage of questions answered exactly correctly
- **F1 Score**: Token-level F1 score for partial matches
- **ROUGE**: Recall-oriented Understudy for Gisting Evaluation
- **BLEU**: Bilingual Evaluation Understudy for machine translation
- **Hallucination Rate**: Percentage of responses with detected hallucinations

### Bootstrap Confidence Intervals

```python
from src.evaluation.bootstrap import BootstrapEvaluator

bootstrap_eval = BootstrapEvaluator(n_samples=1000)
confidence_intervals = bootstrap_eval.compute_confidence_intervals(
    scores=your_scores,
    confidence_level=0.95
)

print(f"95% CI: [{confidence_intervals[0]:.3f}, {confidence_intervals[1]:.3f}]")
```

## Advanced Usage

### Custom Embedding Models

```python
from sentence_transformers import SentenceTransformer
from src.retrieval.dense_retriever import DenseRetriever

# Load custom model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
retriever = DenseRetriever(model=model)

# Use in hybrid setup
hybrid = HybridRetriever(
    sparse_retriever=BM25Retriever(),
    dense_retriever=retriever,
    sparse_weight=0.4,
    dense_weight=0.6
)
```

### Batch Processing

```python
# Process multiple queries efficiently
queries = [
    "What is machine learning?",
    "Explain neural networks",
    "What is natural language processing?"
]

batch_results = retriever.batch_retrieve(queries, corpus, batch_size=8)

for query, results in zip(queries, batch_results):
    print(f"Query: {query}")
    print(f"Top result: {results[0]['text'][:100]}...")
    print()
```

### Custom Document Processing

```python
from src.utils.document_processor import DocumentProcessor

processor = DocumentProcessor()

# Process different file types
documents = processor.process_files([
    "document.pdf",
    "article.txt",
    "webpage.html"
])

# Extract metadata
for doc in documents:
    print(f"Title: {doc['title']}")
    print(f"Content length: {len(doc['content'])}")
    print(f"Source: {doc['source']}")
```

### Performance Monitoring

```python
import time
from src.utils.performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor()

with monitor.track_operation("retrieval"):
    results = retriever.retrieve(query, corpus)

print(f"Retrieval time: {monitor.get_last_duration():.3f}s")
print(f"Memory usage: {monitor.get_memory_usage():.2f}MB")
```

## Troubleshooting

### Common Issues

#### Slow Performance

- Reduce `TOP_K` in configuration
- Use smaller embedding models
- Enable GPU acceleration if available
- Process documents in smaller batches

#### Memory Errors

- Reduce `BATCH_SIZE` in configuration
- Process fewer documents at once
- Use CPU-only mode for large datasets
- Clear cache periodically

#### Low Quality Results

- Adjust sparse/dense weights
- Try different embedding models
- Increase `TOP_K` for more candidates
- Fine-tune generation parameters

### Getting Help

- Check the [GitHub Issues](https://github.com/username/repo/issues) page
- Review the [Architecture Guide](architecture.md) for technical details
- Run the test suite: `python -m pytest tests/`
- Enable debug logging: `export LOG_LEVEL=DEBUG`
