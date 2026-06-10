# ML Tutor - RAG-Based Learning Assistant

A **Retrieval-Augmented Generation (RAG)** powered learning assistant that answers questions about machine learning concepts from "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow". The system uses vector embeddings and local LLM inference to provide accurate, context-aware answers backed by source material.

## 🎯 Project Overview

ML Tutor is an intelligent Q&A system that processes an ML textbook PDF and enables semantic search combined with generative AI. When you ask a question, the system:

1. **Retrieves** relevant content from the document using vector similarity search
2. **Augments** the retrieved context with the original question
3. **Generates** coherent, accurate answers using a local LLM (Ollama)

This RAG approach ensures responses are grounded in the actual textbook content, reducing hallucinations and providing reliable learning assistance.

### Key Features

- **📚 Automatic PDF Indexing**: Loads and chunks PDF documents into a vector database on startup
- **🔍 Semantic Search**: Retrieves relevant sections using semantic similarity (no keyword matching)
- **🤖 Local LLM Inference**: Uses Ollama with Llama 3.2 for private, on-device LLM inference
- **⚡ Fast Retrieval**: Vector similarity search with configurable top-k results
- **💬 RESTful API**: FastAPI backend with interactive Swagger UI for easy querying
- **📖 Source Attribution**: Optionally includes source document references in responses
- **🔧 Persistent Vector Store**: ChromaDB for efficient storage and reuse of embeddings

## 🛠️ Technology Stack

### Backend
- **Python 3.13+**: Core language
- **FastAPI**: Modern web framework for API
- **LangChain**: Framework for building RAG chains and LLM interactions
- **ChromaDB**: Vector database for storing and searching embeddings
- **Sentence Transformers** (`all-MiniLM-L6-v2`): Open-source embedding model
- **PyPDF**: PDF document loading and parsing
- **Ollama**: Local LLM runtime (Llama 3.2)
- **Uvicorn**: ASGI server

## 📋 Prerequisites

- **Python 3.13+**
- **Ollama**: Download from [ollama.ai](https://ollama.ai) and run `ollama pull llama3.2`
- **PDF Document**: Place your `dataset.pdf` in the `dataset/` directory
- **Virtual Environment**: Recommended for dependency isolation

## 🚀 Quick Start

### 1. Clone and Setup

```bash
cd /home/abhinavmishra/Desktop/ML-Tutor
source .venv/bin/activate  # Activate virtual environment
```

### 2. Install Dependencies

```bash
pip install -e .
```

Or using the requirements file:

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file in the root directory:

```env
# LLM Configuration
LLM_API_BASE_URL=http://localhost:11434
LLM_API_KEY=ollama
LLM_MODEL_NAME=llama3.2
LLM_TEMPERATURE=0.1

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Document Configuration
PDF_PATH=./dataset/dataset.pdf

# Vector Store Configuration
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION_NAME=Hands on Machine Learning
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Query Configuration
DEFAULT_TOP_K=4
MAX_TOP_K=10
```

### 4. Ensure Ollama is Running

```bash
ollama serve
# In another terminal, verify: ollama list
```

### 5. Start the FastAPI Server

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`
- **API Docs**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### 6. Query the API

```bash
curl -X POST "http://localhost:8000/api/v1/query/" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is gradient descent?", "top_k": 4, "include_source": true}'
```

## 📁 Project Structure

```
ML-Tutor/
├── main.py                     # Entry point (minimal wrapper)
├── pyproject.toml              # Project metadata and dependencies
├── requirements.txt            # Pip requirements
├── README.md                   # This file
│
├── app/                        # Main application package
│   ├── main.py                # FastAPI application setup and lifespan
│   │
│   ├── api/                   # API layer
│   │   ├── __init__.py
│   │   ├── dependencies.py    # Dependency injection (FastAPI Depends)
│   │   └── routes/
│   │       ├── __init__.py
│   │       └── query.py       # /api/v1/query endpoint
│   │
│   ├── core/                  # Core application configuration
│   │   ├── __init__.py
│   │   ├── config.py          # Settings via Pydantic (env variables)
│   │   ├── exceptions.py      # Custom exception classes
│   │   └── logging_config.py  # Logging setup
│   │
│   ├── models/                # Data models and schemas
│   │   ├── __init__.py
│   │   └── schemas.py         # Pydantic models for request/response
│   │
│   ├── services/              # Business logic layer
│   │   ├── __init__.py
│   │   ├── document_loader.py # PDF loading and chunking
│   │   ├── embedding_service.py # Text embedding generation
│   │   ├── vector_store.py    # ChromaDB interactions
│   │   └── rag_chain.py       # RAG pipeline (retrieval + generation)
│   │
│   └── utils/                 # Utility functions
│       └── __init__.py
│
├── dataset/                   # Document storage
│   └── dataset.pdf           # Place your PDF here
│
└── chroma_db/                # Vector database (auto-created)
    ├── chroma.sqlite3
    └── [collection data]/
```

**Note**: This is a backend-only project. The API can be accessed via:
- Direct HTTP requests (curl, Python, etc.)
- Interactive Swagger UI at `http://localhost:8000/docs`
- ReDoc documentation at `http://localhost:8000/redoc`

## 🔄 How It Works

### 1. **Initialization (Startup)**
- `app/main.py` defines FastAPI lifespan events
- On startup, if ChromaDB doesn't have documents:
  - `DocumentLoader` reads the PDF and chunks it (1000 tokens, 200 overlap)
  - `EmbeddingService` generates embeddings using Sentence Transformers
  - `VectorStoreService` stores chunks in ChromaDB

### 2. **Query Processing (Runtime)**
```
User Query → FastAPI Endpoint → RAGChainService
  ↓
VectorStore.query() → Retrieve top-k similar chunks
  ↓
ChatOllama (Llama 3.2) → Generate answer from context
  ↓
QueryResponse → Return to client
```

## 📝 API Reference

### Query Endpoint

**POST** `/api/v1/query/`

**Request:**
```json
{
  "query": "What is overfitting?",
  "top_k": 4,
  "include_source": true
}
```

**Response:**
```json
{
  "query": "What is overfitting?",
  "answer": "Overfitting occurs when a model learns the training data too well...",
  "sources": [
    {
      "content": "A model that overfits the training data...",
      "metadata": {"page": 5}
    }
  ],
  "retrieval_time_ms": 45,
  "generation_time_ms": 320
}
```

## 🔧 Configuration Details

All settings are loaded from `.env` file via `app/core/config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_BASE_URL` | http://localhost:11434 | Ollama API endpoint |
| `LLM_MODEL_NAME` | llama3.2 | Model to use |
| `LLM_TEMPERATURE` | 0.1 | Generation temperature (lower = more deterministic) |
| `EMBEDDING_MODEL` | sentence-transformers/all-MiniLM-L6-v2 | Embedding model |
| `PDF_PATH` | ./dataset/dataset.pdf | Path to input PDF |
| `CHUNK_SIZE` | 1000 | Tokens per document chunk |
| `CHUNK_OVERLAP` | 200 | Token overlap between chunks |
| `DEFAULT_TOP_K` | 4 | Default number of retrieved chunks |
| `MAX_TOP_K` | 10 | Maximum retrievable chunks |

## 🧪 Testing

### Test with Python

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/query/",
    json={"query": "Explain linear regression", "top_k": 4}
)
print(response.json())
```

### Interactive API Explorer

Open `http://localhost:8000/docs` in your browser for an interactive Swagger UI.

## 🚨 Troubleshooting

**Problem: "PDF file not found at ./dataset/dataset.pdf"**
- Solution: Ensure your PDF is placed at `dataset/dataset.pdf`

**Problem: Ollama connection refused**
- Solution: Start Ollama with `ollama serve` in another terminal

**Problem: Embeddings are slow on first startup**
- Solution: Normal on first run. Embeddings are cached in ChromaDB after initial indexing

**Problem: "Model llama3.2 not found"**
- Solution: Run `ollama pull llama3.2` to download the model

## 📚 Key Files and Responsibilities

| File | Purpose |
|------|---------|
| [app/main.py](app/main.py) | FastAPI app setup, lifespan hooks |
| [app/api/routes/query.py](app/api/routes/query.py) | Query endpoint handler |
| [app/services/document_loader.py](app/services/document_loader.py) | PDF parsing and chunking |
| [app/services/embedding_service.py](app/services/embedding_service.py) | Text embedding generation |
| [app/services/vector_store.py](app/services/vector_store.py) | ChromaDB operations |
| [app/services/rag_chain.py](app/services/rag_chain.py) | RAG pipeline orchestration |
| [app/core/config.py](app/core/config.py) | Configuration management |
| [app/core/exceptions.py](app/core/exceptions.py) | Custom exceptions |
| [app/models/schemas.py](app/models/schemas.py) | Pydantic request/response models |

## � Usage

### Access the API

**Option 1: Interactive Swagger UI**
```
http://localhost:8000/docs
```

**Option 2: cURL Request**
```bash
curl -X POST "http://localhost:8000/api/v1/query/" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is gradient descent?", "top_k": 4, "include_source": true}'
```

**Option 3: Python Script**
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/query/",
    json={"query": "Explain linear regression", "top_k": 4}
)
print(response.json())
```

### API Endpoints

#### Query the RAG System
```
POST /api/v1/query/
```

**Request Body:**
```json
{
  "query": "What is overfitting?",
  "top_k": 4,
  "include_source": true
}
```

**Response:**
```json
{
  "query": "What is overfitting?",
  "answer": "Overfitting occurs when a model learns the training data too well...",
  "sources": [
    {
      "content": "A model that overfits the training data...",
      "metadata": {"page": 5}
    }
  ],
  "retrieval_time_ms": 45,
  "generation_time_ms": 320
}
```

## 🔄 How the RAG Pipeline Works

1. **Query Embedding**: Your question is converted to a vector embedding
2. **Semantic Retrieval**: The system finds the most relevant sections from the PDF using vector similarity
3. **Context Augmentation**: Retrieved chunks are combined with your original question
4. **Answer Generation**: The LLM (Llama 3.2 via Ollama) generates an answer based on the context
5. **Response**: The answer and source references are returned to you

## 📚 Sample Use Cases

- "Explain overfitting and how to prevent it"
- "What's the difference between SGD and batch gradient descent?"
- "How do I implement a decision tree in scikit-learn?"
- "What are neural network activation functions?"
- "Explain cross-validation"

## 🐛 Troubleshooting

### Chroma Database Issues
If you want to reset and re-index from scratch:
```bash
rm -rf ./chroma_db
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### API Connection Issues
- Verify backend is running: `curl http://localhost:8000/docs`
- Check that Ollama is running: `ollama serve`
- Ensure the PDF is placed at `dataset/dataset.pdf`

### Embedding Generation Issues
- First startup takes longer due to embedding generation (cached after initial run)
- Ensure you have the Sentence Transformers model downloaded (auto-downloads on first use)
- Check that you have sufficient disk space for ChromaDB

### Common Errors
- **"PDF file not found"**: Place your PDF at `dataset/dataset.pdf`
- **"Ollama connection refused"**: Run `ollama serve` in another terminal
- **"Model llama3.2 not found"**: Run `ollama pull llama3.2`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Resources

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Ollama Documentation](https://ollama.ai/)
- [Sentence Transformers](https://www.sbert.net/)
- [Hands-On Machine Learning Book](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/)

## 📞 Support

For questions or issues:
- Open a GitHub issue
- Review this README for setup and configuration help
- Check the FastAPI interactive docs at `/docs` for endpoint details

---

**Happy Learning! 🚀**
