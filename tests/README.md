# E2E Tests for RAG-Based Chatbot

This directory contains end-to-end tests for the query endpoints.

## Prerequisites

1. **Install test dependencies:**
   ```bash
   pip install ".[dev]"
   ```

2. **Set up environment variables:**
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```

3. **Ensure vectorstore exists (only if not already built):**
   ```bash
   python src/build_index.py
   ```

   This creates the `chroma_db` directory with indexed FAQ data. **Only run this if the vectorstore doesn't already exist.** Tests will automatically use the existing vectorstore if it's available, or skip if it's empty.

## Running Tests

### Run all tests:
```bash
pytest tests/
```

### Run specific test file:
```bash
pytest tests/test_query_endpoints.py
```

### Run specific test class:
```bash
pytest tests/test_query_endpoints.py::TestQueryEndpoint
```

### Run with verbose output:
```bash
pytest tests/ -v
```

### Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## Test Structure

### `conftest.py`
Contains pytest fixtures:
- `temp_dir`: Temporary directory for test files
- `temp_cache_dir`: Temporary cache directory (overrides cache path)
- `temp_metrics_dir`: Temporary metrics directory (overrides metrics path)
- `temp_vectorstore_dir`: Temporary vectorstore directory (overrides vectorstore path)
- `test_client`: FastAPI TestClient instance
- `openai_api_key`: OpenAI API key from environment (skips tests if not set)
- `sample_question`: Sample question request
- `sample_question_with_filters`: Sample question with filters

### `test_query_endpoints.py`
Contains test classes:

1. **`TestQueryEndpoint`**: Tests for `POST /api/v1/query/`
   - Basic query without filters
   - Query with metadata filters
   - Caching behavior
   - Input validation
   - Missing API key handling

2. **`TestMetricsEndpoint`**: Tests for `GET /api/v1/query/metrics`
   - Empty metrics (no requests)
   - Metrics after making queries
   - Metrics structure validation

3. **`TestEvaluateEndpoint`**: Tests for `GET /api/v1/query/evaluate/{query_id}`
   - Non-existent query_id
   - Successful evaluation
   - Incomplete cached data

4. **`TestEndpointsIntegration`**: Integration tests
   - Full workflow: query -> metrics -> evaluate
   - Caching verification

## Test Isolation

Each test function uses temporary directories for:
- Cache files (`cache/cache.json`)
- Metrics files (`metrics/metrics.json`)
- Vectorstore (optional, can use real one)

This ensures tests don't interfere with each other or with production data.

## Notes

- Tests that require OpenAI API will be skipped if `OPENAI_API_KEY` is not set
- Tests that require a vectorstore will be skipped if the vectorstore is empty
- The `temp_vectorstore_dir` fixture creates an empty directory by default. To use the real vectorstore, uncomment the copy logic in the fixture
- Rate limiting is disabled in tests (TestClient doesn't trigger rate limiters)

## Troubleshooting

### Tests are skipped
- Check if `OPENAI_API_KEY` is set
- Check if vectorstore exists and has data (`./chroma_db`)

### Import errors
- Make sure you're running tests from the project root
- Ensure `src/` is in Python path (handled by `src/main.py`)

### Cache/metrics not isolated
- Ensure fixtures are properly overriding module-level constants
- Check that each test uses the fixtures correctly

