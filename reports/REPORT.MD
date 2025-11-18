# 📘 AEM2PI — RAG FAQ Support Chatbot

By Jerry Chibuokem

## ✅ TL;DR

In this project, I built a production-grade RAG FAQ chatbot that:

- ✅ parses a document with structured metadata
- ✅ uses smart recursive, section-based chunking
- ✅ generates embeddings with OpenAI
- ✅ uses ChromaDB + cosine similarity + ANN
- ✅ performs hybrid retrieval (vector + metadata filtering)
- ✅ returns structured JSON via FastAPI REST API
- ✅ tracks latency, cost, and reliability metrics
- ✅ includes query caching + comprehensive error handling
- ✅ provides a clean indexing & query pipeline
- ✅ includes an evaluator agent for answer quality assessment

This approach strikes the best balance between:

- ✔ accuracy
- ✔ cost efficiency
- ✔ performance
- ✔ clean architecture
- ✔ observability

## 1. Context & Objective

This project was built for **HRCare** — an HR SaaS that helps organizations manage hiring, onboarding, payroll integrations, time-off, performance, compliance, and employee lifecycle workflows.

Our customer support team receives large volumes of repetitive questions daily — all of which already exist in long-form internal documentation. My objective in this project was to build a Retrieval-Augmented Generation (RAG) system capable of:

- parsing a long FAQ document
- intelligently chunking the content
- generating vector embeddings
- storing them in a vector DB
- retrieving the most relevant chunks for any new question
- and generating final answers with full transparency

This system returns structured JSON responses via a FastAPI REST API with endpoints for querying, metrics tracking, and answer evaluation.

## 2. Tools & Libraries Used

I used the LangChain RAG ecosystem:

- `langchain`
- `langchain-community`
- `langchain-openai`
- `langchain-chroma`
- `fastapi` (for the REST API)
- `python-dotenv` (for environment variable management)
- `tiktoken` (for accurate token counting)

### Embeddings

I used OpenAI's embedding models (default: `text-embedding-ada-002`, with support for `text-embedding-3-small` and `text-embedding-3-large`) because they balance cost, speed, and quality.

## 3. Document Processing & Chunking Strategy

### 🔹 Recursive (Section-Based) Chunking 

My FAQ document is naturally structured into sections such as:

- SSO
- Billing
- Payroll
- ATS
- Policies

These sections are already semantically meaningful, so instead of chunking the raw text blindly, I first parsed each FAQ entry into its own metadata bundle:

```json
{
  "section_id": "...",
  "section": "...",
  "product_area": "...",
  "intent_tags": [...],
  "content": "..."
}
```

Then I applied chunking inside each **section only**.

### 🔹 Why This Makes Sense

- Sections are semantically coherent
- I avoid mixing unrelated topics
- Retrieval becomes more accurate
- No unnecessary duplication across sections from sliding windows
- No random fragmentation from fixed-size chunking

### 🔹 RecursiveCharacterTextSplitter

I used a hierarchical separator list:

```python
separators = ["\n\n", "\n", ". ", " ", ""]
```

This attempts to split at the largest natural boundary, then progressively falls back to smaller units:

1. paragraph
2. line
3. sentence
4. word
5. characters

**Target chunk size:**
- ~100 words (≈ 600 characters)
- ~20-word overlap

Many sections were already short, so the splitter kept them intact.

### 🔹 Result

I produced coherent, topic-aligned chunks, ideal for high-quality RAG retrieval.

## 4. Vector Store & Similarity Search

### 🔹 Vector DB: Chroma

Used ChromaDB from `langchain-chroma`.

**My choice: Cosine similarity**

I configured Chroma's HNSW index to use cosine:

```python
collection_metadata={"hnsw:space": "cosine"}
```

**Why choose Cosine over Euclidean (Chroma's default)?**

| Metric | Weakness | Why I Rejected It |
|--------|----------|-------------------|
| L2 (Euclidean) | Measures spatial distance | Measures straight-line distance between points, which doesn't align well with semantic similarity. For text embeddings, I wasn't interested in the magnitude of vectors, I wanted directional similarity (angle). |
| Cosine | Magnitude-invariant | Focuses on the angle/direction between vectors, which better captures semantic similarity for text embeddings. |

**Note:** Cosine similarity measures directional alignment (angle), not spatial distance, making it more appropriate for semantic similarity in text embeddings.

### 🔹 Cosine Distance → Similarity

Chroma does not return cosine similarity directly.
It returns cosine distance (0 = identical, higher = worse).

Example:
- 0.12 → highly similar (similarity = 0.88)
- 0.23 → less similar (similarity = 0.77)
- 0.45 → weak similarity (similarity = 0.55)

### 🔹 ANN (Approximate Nearest Neighbor)

Using HNSW provides:
- sub-millisecond lookups
- scales well for large corpora
- excellent recall for RAG systems

## 5. Retrieval Enhancements

### 🔹 1. Score Thresholding

I implemented a configurable cosine similarity threshold (default: 0.78).
Chunks with similarity below this threshold are discarded.

This prevents irrelevant context from being sent to the LLM, thereby wasting tokens.

### 🔹 2. Hybrid Search

I implemented a hybrid retrieval approach:
- **Vector search** (semantic meaning via cosine similarity)
- **Metadata filtering** (exact matches on `section`, `product_area`, etc.)

Hybrid search improves:
- exact term recall
- acronym matching (SSO, MFA)
- highly specific policy queries

The system uses Chroma's `similarity_search_with_score` with optional `filter` parameter for metadata-based filtering.

## 6. Query Pipeline

For each question via the `/api/v1/query/` endpoint:

1. Validate question (max 100 words, Pydantic validation)
2. Check cache for exact question match
3. Embed the question (with token and dimension validation)
4. Perform vector search (ANN + optional metadata filters)
5. Filter via similarity threshold
6. Construct context from retrieved chunks
7. Validate and truncate context if it exceeds LLM token limits
8. Send context to LLM with anti-hallucination instructions
9. Return structured JSON response with:
   - `query_id` (hash of question)
   - `answer`
   - `context_used` (full context chunks with metadata)
   - `sources` (chunk IDs)
   - `no_context_found` flag

## 7. Error Handling & Validations

### ✔ Embedding Input Validation

- Validate token count (max tokens per embedding model)
- Truncate if too long (prevent OpenAI embedding error)
- Ensure vector dimension matches the Chroma store dimension

### ✔ Generation Safety

- Validate prompt token count using `tiktoken`
- Truncate long contexts if they exceed LLM token limits
- Strip invalid characters

### ✔ Query-Time Error Handling

- Retry logic for OpenAI failures
- Graceful fallback when no chunks match
- Structured error messages
- Catch corrupted files, missing metadata, invalid UTF-8

### ✔ Indexing Error Handling

- Handle malformed sections
- Skip empty content blocks
- Ensure minimum chunk quality

## 8. Performance, Observability & Costs

### 📉 Latency Metrics

Tracked via `/api/v1/query/metrics` endpoint:
- `avgLatency` (average response time in ms)
- `p50Latency` (median response time)
- `p95Latency` (95th percentile)
- `throughput` (requests per second)
- `errors & successes`

### 💰 Cost Tracking

I track and report:
- `totalEmbeddingCost` (cost for generating embeddings)
- `totalLlmCost` (cost for LLM generation)
- `totalCost` (total cost per query)
- Cumulative cost over all requests
- Per-request cost breakdown in `recent` requests

Costs are calculated from actual API usage (token counts) using current OpenAI pricing.

### 📈 Reliability

- Comprehensive error handling with custom exceptions
- Graceful degradation when services are unavailable
- Structured error responses

### 🧊 Caching

I implemented:
- **Query Cache** → prevents repeat embedding and LLM calls for identical questions
- Cache stored in `cache/cache.json` (max 100 entries)
- Cache lookup by question hash (`query_id`)
- Metrics are not re-recorded for cached responses

## 9. Design Tradeoffs

- **Latency**: Cosine + ANN gives extremely fast search
- **Throughput**: Efficient embedding and caching keeps response times low
- **Cost**: Small chunk sizes and caching reduce overall token usage
- **Accuracy**: Section-based chunking avoids topic mixing
- **Transparency**: Full context returned in responses for evaluation

## 10. How to Run the Indexing Pipeline

### 1. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install .
```

### 2. Set your environment variables

Create `.env` file:

```bash
OPENAI_API_KEY=your-key
OPENAI_API_BASE=your-base-url  # Optional, for open router
```

### 3. Run the indexing script

```bash
python src/build_index.py
```

This will:
- load the FAQ file (`data/faq_document.txt`)
- parse + chunk
- embed
- populate Chroma vector store (`./chroma_db`)

### 4. Start the FastAPI server

```bash
fastapi dev src/main.py
```

The server will be available at `http://localhost:8000`

- **API Docs**: http://localhost:8000/docs

### 5. Query the API

**Query endpoint:**
```bash
POST /api/v1/query/
{
  "question": "How do I reset my SSO credentials?",
  "filters": {
    "section": "Account & Access",
    "product_area": "Account Management"
  },
  "min_similarity": 0.78
}
```

**Response:**
```json
{
  "query_id": "a1b2c3d4e5f6...",
  "answer": "...",
  "context_used": [
    {
      "content": "...",
      "section_id": "ACCOUNT_Q1_CREATE",
      "section": "Account & Access",
      "similarity_score": 0.88
    }
  ],
  "sources": ["ACCOUNT_Q1_CREATE_chunk_1"],
  "no_context_found": false
}
```

**Metrics endpoint:**
```bash
GET /api/v1/query/metrics
```

**Evaluation endpoint:**
```bash
GET /api/v1/query/evaluate/{query_id}
```

## 11. Bonus: Evaluator Agent

I implemented an evaluation agent at `/api/v1/query/evaluate/{query_id}`:

- Takes a `query_id` from cache
- Extracts the question, answer, and context used
- Makes another LLM call to verify no hallucination in the response
- Returns:
  - `verdict`: `RELIABLE` or `SUSPECTED_HALLUCINATION`
  - `confidence`: 0.0 to 1.0
  - `possible_hallucination`: `true` if confidence < 0.78
  - `reasoning`: Brief explanation of the evaluation

This provides programmatic quality assessment of generated answers.

