# ML Tutor - RAG-Based Learning Assistant

A Retrieval-Augmented Generation (RAG) powered learning assistant that answers questions about machine learning concepts from "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" using vector search and LLMs.

## 🎯 Project Overview

ML Tutor combines a **React frontend** with a **RAG backend** to create an intelligent Q&A system. The app indexes content from the ML textbook into a Chroma vector database, enabling semantic search and context-aware answers powered by large language models.

### Key Features

- **📚 Smart Content Indexing**: Automatically processes and embeds ML textbook content into Chroma vector database
- **🔍 Semantic Search**: Find relevant content using natural language queries
- **🤖 RAG-Powered Responses**: Generate accurate, context-aware answers backed by source material
- **⚡ Fast Retrieval**: Vector similarity search for instant contextual information
- **💬 Interactive Chat UI**: Clean React interface for asking questions and exploring ML concepts
- **📖 Source Attribution**: Responses include references to the source book sections

## 🛠️ Technology Stack

### Backend
- **Python 3.9+**
- **ChromaDB**: Vector database for storing and retrieving embeddings
- **LangChain**: Framework for building RAG pipelines
- **FastAPI** or **Flask**: REST API server
- **Scikit-learn & TensorFlow**: ML library references
- **Sentence Transformers/OpenAI Embeddings**: For vectorizing text

### Frontend
- **React 18+**: UI framework
- **TypeScript**: Type safety
- **Axios/Fetch**: API communication
- **Tailwind CSS** (optional): Styling

## 📋 Prerequisites

- Python 3.9 or higher
- Node.js 16+ and npm/yarn
- OpenAI API key or other LLM provider credentials (if using external LLM)
- Git

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd ML-tutor
```

### 2. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your API keys and configuration
```

### 3. Frontend Setup

```bash
# Navigate to frontend directory
cd ../frontend

# Install dependencies
npm install

# Create environment file
cp .env.example .env.local
# Edit .env.local with API endpoint
```

## 📁 Project Structure

```
ML-tutor/
├── backend/
│   ├── app.py                 # Main FastAPI/Flask app
│   ├── config.py              # Configuration settings
│   ├── requirements.txt        # Python dependencies
│   ├── .env.example           # Environment variables template
│   ├── src/
│   │   ├── ingestion/         # Data indexing pipeline
│   │   │   └── book_loader.py # Load and process textbook
│   │   ├── rag/               # RAG pipeline
│   │   │   ├── retriever.py   # Vector search logic
│   │   │   └── generator.py   # LLM response generation
│   │   ├── api/               # API endpoints
│   │   │   └── routes.py      # FastAPI routes
│   │   └── utils/             # Utility functions
│   └── data/
│       └── chroma_db/         # Local Chroma database storage
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/        # React components
│   │   │   ├── ChatBox.tsx
│   │   │   ├── Message.tsx
│   │   │   └── SourceCitation.tsx
│   │   ├── pages/
│   │   │   └── Home.tsx
│   │   ├── services/          # API service layer
│   │   │   └── api.ts
│   │   ├── App.tsx
│   │   └── index.tsx
│   ├── package.json
│   ├── tsconfig.json
│   └── .env.example
│
├── docs/                      # Documentation
│   ├── SETUP.md              # Detailed setup guide
│   └── API.md                # API documentation
│
├── .gitignore
└── README.md
```

## 🔧 Configuration

### Backend (.env)
```env
# LLM Configuration
LLM_PROVIDER=openai              # or: ollama, huggingface
OPENAI_API_KEY=your_api_key
LLM_MODEL=gpt-3.5-turbo

# Embeddings Configuration
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_PROVIDER=openai

# Database Configuration
CHROMA_DB_PATH=./data/chroma_db
CHROMA_COLLECTION_NAME=ml_textbook

# Server Configuration
API_PORT=8000
API_HOST=0.0.0.0

# RAG Configuration
RETRIEVAL_K=5                   # Number of documents to retrieve
TEMPERATURE=0.7                 # LLM temperature
```

### Frontend (.env.local)
```env
REACT_APP_API_URL=http://localhost:8000
REACT_APP_API_BASE=/api/v1
```

## 💻 Usage

### Start the Backend

```bash
cd backend
source venv/bin/activate
python app.py
```

The API will be available at `http://localhost:8000`

### Start the Frontend

```bash
cd frontend
npm start
```

The app will open at `http://localhost:3000`

### API Endpoints

#### Ask a Question
```bash
POST /api/v1/ask
Content-Type: application/json

{
  "question": "What is gradient descent?"
}
```

Response:
```json
{
  "answer": "Gradient descent is an optimization algorithm...",
  "sources": [
    {
      "section": "Chapter 4: Gradient Descent",
      "page": 125,
      "excerpt": "..."
    }
  ]
}
```

#### Index/Upload Content
```bash
POST /api/v1/index
Content-Type: multipart/form-data

file: <pdf_or_text_file>
```

## 🎓 Workflow

1. **Ingestion Phase**
   - Load textbook content (PDF/text)
   - Split into chunks
   - Generate embeddings using Sentence Transformers or OpenAI API
   - Store in Chroma vector database

2. **Query Phase**
   - User asks a question in the React UI
   - Backend receives the query
   - Convert query to embeddings
   - Retrieve similar documents from Chroma (top-k retrieval)
   - Pass retrieved context + query to LLM
   - LLM generates answer with citations

3. **Response Phase**
   - Return answer and sources to frontend
   - Display with proper formatting and source links

## 📚 Sample Use Cases

- "Explain overfitting and how to prevent it"
- "What's the difference between SGD and batch gradient descent?"
- "How do I implement a decision tree in scikit-learn?"
- "What are neural network activation functions?"
- "Explain cross-validation"

## 🐛 Troubleshooting

### Chroma Database Issues
```bash
# Reset Chroma database
rm -rf backend/data/chroma_db
# Re-index the content
python backend/src/ingestion/book_loader.py
```

### API Connection Issues
- Verify backend is running: `curl http://localhost:8000/health`
- Check frontend `.env.local` has correct API URL
- Look for CORS configuration in backend

### Embedding Generation Issues
- Verify API keys in `.env`
- Check internet connection for external LLM providers
- For local models, ensure sufficient memory

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
- [React Documentation](https://react.dev/)
- [Hands-On Machine Learning Book](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/)

## 📞 Support

For questions or issues:
- Open an GitHub issue
- Check existing documentation in `/docs`
- Review API documentation in `/docs/API.md`

---

**Happy Learning! 🚀**
