"""FastAPI application entry point."""

import os
import sys
from pathlib import Path

# Add src directory to Python path BEFORE any other imports
src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from querying.router import router as query_router

app = FastAPI(
    title="RAG-Based Chatbot",
    version="0.1.0",
    description="A RAG-based chatbot service with question answering",
)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Make limiter available to routers
def setup_limiter():
    import querying.router
    querying.router.limiter = limiter

setup_limiter()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
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


@app.head("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}
