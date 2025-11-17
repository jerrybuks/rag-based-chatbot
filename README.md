# RAG-Based Chatbot

A RAG-based chatbot service built with FastAPI.

## Setup

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -e ".[dev]"
```

## Running the Application

```bash
uvicorn src.main:app --reload
```

The server will be available at `http://localhost:8000`

- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
