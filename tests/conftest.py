"""Pytest configuration and fixtures for e2e tests."""

import os
import sys
import json
import shutil
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from main import app


@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture(scope="function")
def temp_cache_dir(temp_dir: Path) -> Generator[Path, None, None]:
    """Create a temporary cache directory."""
    cache_dir = temp_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "cache.json"
    cache_file.write_text(json.dumps({}), encoding="utf-8")
    
    # Temporarily override cache path
    import querying.cache as cache_module
    original_cache_file = cache_module.CACHE_FILE
    cache_module.CACHE_FILE = cache_file
    
    yield cache_dir
    
    # Restore original
    cache_module.CACHE_FILE = original_cache_file


@pytest.fixture(scope="function")
def temp_metrics_dir(temp_dir: Path) -> Generator[Path, None, None]:
    """Create a temporary metrics directory."""
    metrics_dir = temp_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = metrics_dir / "metrics.json"
    metrics_file.write_text(
        json.dumps({"requests": [], "successes": 0, "failures": 0}),
        encoding="utf-8"
    )
    
    # Temporarily override metrics path
    import querying.metrics as metrics_module
    original_metrics_file = metrics_module.METRICS_FILE
    metrics_module.METRICS_FILE = metrics_file
    
    yield metrics_dir
    
    # Restore original
    metrics_module.METRICS_FILE = original_metrics_file


@pytest.fixture(scope="function")
def temp_vectorstore_dir(temp_dir: Path) -> Generator[Path, None, None]:
    """Create a temporary vectorstore directory.
    
    Copies the real vectorstore if it exists, otherwise creates an empty directory.
    """
    vectorstore_dir = temp_dir / "chroma_db"
    vectorstore_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy existing vectorstore for testing if it exists
    real_vectorstore = Path("./chroma_db")
    if real_vectorstore.exists() and any(real_vectorstore.iterdir()):
        shutil.copytree(real_vectorstore, vectorstore_dir, dirs_exist_ok=True)
    
    # Temporarily override vectorstore path
    # The app already loads the router module, so access it from sys.modules
    # This gets the actual module, not the router object exported by querying.__init__.py
    if 'querying.router' not in sys.modules:
        # Force load the module if not already loaded
        import importlib
        importlib.import_module("querying.router")
    
    router_module = sys.modules['querying.router']
    original_vectorstore_path = router_module.DEFAULT_VECTORSTORE_PATH
    router_module.DEFAULT_VECTORSTORE_PATH = vectorstore_dir
    
    yield vectorstore_dir
    
    # Restore original
    router_module.DEFAULT_VECTORSTORE_PATH = original_vectorstore_path


@pytest.fixture(scope="function")
def test_client(
    temp_cache_dir: Path,
    temp_metrics_dir: Path,
    temp_vectorstore_dir: Path,
) -> Generator[TestClient, None, None]:
    """Create a test client for the FastAPI app."""
    # Note: The app uses the actual vectorstore, cache, and metrics paths
    # which are overridden by the fixtures above
    client = TestClient(app)
    yield client


@pytest.fixture(scope="function")
def openai_api_key() -> str:
    """Get OpenAI API key from environment or skip test if not available."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set. Skipping tests that require OpenAI API.")
    return api_key


@pytest.fixture(scope="function")
def sample_question() -> dict:
    """Sample question request for testing."""
    return {
        "question": "How do I create an account?",
        "min_similarity": 0.78
    }


@pytest.fixture(scope="function")
def sample_question_with_filters() -> dict:
    """Sample question request with filters for testing."""
    return {
        "question": "How do I create an account?",
        "filters": {
            "section": "Account & Access",
            "product_area": "Account Management"
        },
        "min_similarity": 0.78
    }

