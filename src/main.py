"""FastAPI application entry point."""

import os
import sys
from pathlib import Path

# Add src directory to Python path BEFORE any other imports
src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from fastapi import FastAPI
from querying.router import router as query_router

app = FastAPI(
    title="RAG-Based Chatbot",
    version="0.1.0",
    description="A RAG-based chatbot service with question answering",
)

# Include query router
app.include_router(query_router)

# Default port configuration
DEFAULT_PORT = int(os.getenv("PORT", 8000))
DEFAULT_HOST = os.getenv("HOST", "0.0.0.0")


@app.get("/")
async def read_root():
    """Root endpoint."""
    return {
        "message": "Welcome to RAG-Based Chatbot",
        "endpoints": {
            "query": "/api/v1/query/",
            "metrics": "/api/v1/query/metrics",
            "health": "/health",
            "docs": "/docs",
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}
