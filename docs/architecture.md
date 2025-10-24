# Architecture Overview

This document describes the technical architecture of the Local Hybrid Retrieval-Augmented Document QA system.

## System Overview

The system implements a hybrid retrieval-augmented generation (RAG) approach that combines sparse and dense retrieval methods to provide accurate, hallucination-resistant question answering over documents.

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │───▶│   Retrieval     │───▶│   Generation    │
│                 │    │   (Hybrid)      │    │   (LLM)         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Hallucination  │    │   Response      │
                       │   Detection     │    │   Filtering     │
                       └─────────────────┘    └─────────────────┘
```

## Core Components

### 1. Retrieval System

#### Hybrid Retriever

Combines sparse (BM25) and dense (embedding-based) retrieval:

```python
class HybridRetriever:
    def __init__(self, sparse_weight=0.3, dense_weight=0.7):
        self.sparse_retriever = BM25Retriever()
        self.dense_retriever = DenseRetriever()
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight

    def retrieve(self, query, corpus, top_k=10):
        # Get sparse and dense scores
        sparse_scores = self.sparse_retriever.score(query, corpus)
        dense_scores = self.dense_retriever.score(query, corpus)

        # Combine scores
        combined_scores = (self.sparse_weight * sparse_scores +
                          self.dense_weight * dense_scores)

        # Return top-k results
        return self._get_top_k(combined_scores, corpus, top_k)
```

#### Sparse Retrieval (BM25)

- **Algorithm**: Best Matching 25
- **Library**: Custom implementation with NLTK
- **Features**:
  - Term frequency-inverse document frequency (TF-IDF)
  - Document length normalization
  - Query term proximity scoring

#### Dense Retrieval (Embeddings)

- **Model**: BGE (`BAAI/bge-base-en-v1.5`)
- **Library**: DocArrayInMemorySearch for vector storage, HuggingFace for embeddings
- **Features**:
  - Semantic similarity matching via cosine distance
  - Contextual understanding through transformer embeddings
  - Local caching in `embed_cache/` for performance
  - GPU acceleration (4.2× speedup on RTX 4050)

### 2. Generation System

#### Local Language Model (Ollama)

- **Platform**: Ollama for local LLM hosting
- **Model**: Llama 3.2 (open-source, fully local)
- **Endpoint**: `http://localhost:11434/api/generate`
- **Features**:
  - 100% local inference (no external API calls)
  - Context-aware response generation with chat history
  - Configurable temperature (0.6), top-p (0.95), top-k (40)
  - GPU acceleration (3× speedup vs CPU)
  - Complete data sovereignty

#### Prompt Engineering

```python
def get_prompt_template():
    return """You are a helpful assistant. Based on the chat history
    and the context provided, answer the user's current question
    in a clear and natural way.

    Chat History:
    {chat_history}

    Context:
    {context}

    Current Question: {question}

    Answer:"""
```

#### Hallucination Detection (Evaluation Only)

- **Method**: LLM-as-Judge approach using Gemini API
- **Note**: Used only for offline evaluation, NOT in production
- **Implementation**: Evaluates faithfulness (1-5), confidence (1-5)
- **Metrics**: Binary hallucination flag, faithfulness score, confidence score

### 3. Client-Server Architecture

#### Server Component (`src/server.py`)

- **Protocol**: TCP socket on port 65432
- **Functions**:
  - Document loading (PDF, CSV, JSON, TXT)
  - Hybrid retrieval orchestration
  - LLM inference via Ollama
  - Chat history management
- **Commands**:
  - `upload`: Initialize retriever with documents
  - `ask`: Query with retrieval and generation
  - `get_files`: List uploaded documents
  - `get_chat_history`: Retrieve conversation history
  - `reset`: Clear all data
  - `get_status`: System health check

#### Client Component (`src/client.py`)

- **Framework**: Flask HTTP API server (port 5001)
- **Role**: Bridges web UI and server via sockets
- **Endpoints**:
  - `POST /rag_upload`: Document upload
  - `POST /rag_ask`: Question answering
  - `GET /rag_get_files`: List documents
  - `GET /rag_get_chat_history`: Conversation history
  - `POST /rag_reset`: System reset
  - `GET /rag_get_status`: Status check

#### Frontend (`frontend/chat.html`)

- **Framework**: Vanilla JavaScript + HTML5
- **Styling**: Custom CSS with responsive design
- **Features**:
  - File upload interface (PDF, CSV, JSON, TXT)
  - Real-time question answering
  - Chat history display
  - Document management

### 4. Evaluation Framework

#### Bootstrap Evaluation

- **Method**: Statistical resampling for confidence intervals
- **Implementation**: Custom BootstrapEvaluator class
- **Metrics**: Exact Match, F1, ROUGE, BLEU, Hallucination Rate

#### Dataset Support

- **SQuAD**: Stanford Question Answering Dataset
- **MS MARCO**: Microsoft Machine Reading Comprehension
- **Natural Questions**: Google search questions

## Data Flow

### Document Processing Pipeline (100% Local)

1. **Document Ingestion**

   ```
   Web Upload → Client (Flask) → Server (Socket) → Document Loaders
   → Text Extraction → RecursiveCharacterTextSplitter (400/150)
   → Chunks with Metadata
   ```

2. **Indexing (Hybrid)**

   ```
   Chunks → BGE Embeddings (GPU/CPU) → DocArrayInMemorySearch (Vector Store)
        ↓
   Chunks → BM25Retriever (Keyword Index)
        ↓
   EnsembleRetriever (30% BM25, 70% Semantic)
   ```

3. **Query Processing (100% Local)**

   ```
   User Question → Client → Server → Hybrid Retrieval (top-5 chunks)
   → Context Assembly → Prompt Template (history + context + question)
   → Ollama/Llama 3.2 (Local LLM) → Answer
   → Chat History Update → Client → Web UI
   ```

4. **Evaluation Pipeline (Offline)**
   ```
   Benchmark Dataset → Hybrid Retrieval → Metrics (MRR, Recall@K)
   → Bootstrap Resampling (1000×) → 95% Confidence Intervals
   → Hallucination Eval (Gemini API) → Faithfulness Scores
   ```

## Configuration Management

### Configuration Classes

```python
@dataclass
class RetrievalConfig:
    sparse_weight: float = 0.3
    dense_weight: float = 0.7
    top_k: int = 10
    embedding_model: str = "all-MiniLM-L6-v2"

@dataclass
class GenerationConfig:
    model_name: str = "microsoft/DialoGPT-small"
    max_new_tokens: int = 100
    temperature: float = 0.7
    do_sample: bool = True

@dataclass
class EvaluationConfig:
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    random_seed: int = 42
```

### Environment Variables

```bash
# Model and data paths
TRANSFORMERS_CACHE=/path/to/models
DATA_DIR=/path/to/datasets

# Performance settings
CUDA_VISIBLE_DEVICES=0
OMP_NUM_THREADS=4

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/system.log
```

## Performance Considerations

### Memory Management

- **Chunking Strategy**: Fixed-size text chunks with overlap
- **Batch Processing**: Configurable batch sizes for GPU/CPU
- **Cache Management**: LRU cache for embeddings and results

### Scalability

- **Vector Database**: FAISS with IVF indexing for large corpora
- **Parallel Processing**: Multi-threading for batch operations
- **Resource Pooling**: Connection pooling for external services

### Optimization Techniques

- **Quantization**: 8-bit quantization for smaller models
- **Pruning**: Model pruning for faster inference
- **Caching**: Multi-level caching (memory, disk, distributed)

## Security and Privacy

### Local Processing

- **No External APIs**: All processing happens locally
- **Data Isolation**: Documents never leave the local system
- **Model Privacy**: Pre-trained models run locally

### Input Validation

- **File Type Checking**: Restrict to safe file formats
- **Size Limits**: Maximum file and corpus sizes
- **Content Filtering**: Basic content validation

## Testing and Quality Assurance

### Unit Tests

- **Coverage**: Target 80%+ code coverage
- **Frameworks**: pytest with fixtures and mocking
- **CI/CD**: GitHub Actions for automated testing

### Integration Tests

- **End-to-End**: Full pipeline testing
- **Performance Tests**: Benchmarking and profiling
- **Load Tests**: Stress testing with large datasets

### Evaluation Metrics

- **Accuracy Metrics**: Exact Match, F1, ROUGE
- **Quality Metrics**: Hallucination detection accuracy
- **Performance Metrics**: Latency, throughput, memory usage

## Deployment Options

### Local Development

```bash
# Development mode
export FLASK_ENV=development
python src/server_MPC.py
```

### Production Deployment

```bash
# Production mode with Gunicorn
gunicorn --bind 0.0.0.0:5000 --workers 4 src.server_MPC:app
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /app
WORKDIR /app

EXPOSE 5000
CMD ["python", "src/server_MPC.py"]
```

## Monitoring and Observability

### Logging

- **Structured Logging**: JSON format with context
- **Log Levels**: DEBUG, INFO, WARNING, ERROR
- **Log Rotation**: Automatic log file rotation

### Metrics

- **Performance Metrics**: Response time, throughput
- **Quality Metrics**: Accuracy, hallucination rates
- **System Metrics**: CPU, memory, disk usage

### Health Checks

- **Endpoint**: `/health` for system status
- **Dependencies**: Check model loading and database connections
- **Readiness**: Ensure system is ready to handle requests

## Future Enhancements

### Planned Features

- **Multi-modal Support**: Images and tables in documents
- **Conversational Memory**: Context retention across queries
- **Model Fine-tuning**: Domain-specific model adaptation
- **Distributed Processing**: Multi-node deployment support

### Research Directions

- **Advanced Retrieval**: Learning-to-rank and re-ranking
- **Better Hallucination Detection**: Multi-model consensus
- **Explainability**: Attention visualization and reasoning traces
- **Multilingual Support**: Cross-language document processing

## References

### Key Papers

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)
- [Measuring and Reducing Hallucinations in Large Language Models](https://arxiv.org/abs/2303.05280)

### Libraries and Tools

- **Transformers**: Hugging Face transformers library
- **FAISS**: Facebook AI Similarity Search
- **Sentence Transformers**: Sentence embedding models
- **Flask**: Web framework for Python

For more technical details, see the source code in the `src/` directory.
