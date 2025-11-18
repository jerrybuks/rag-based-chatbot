# RAG-Based Chatbot

A RAG-based chatbot service built with FastAPI.

## Live Demo & Monitoring

- **Product Demo**: [https://rag-chatbot-fe-self.vercel.app/](https://rag-chatbot-fe-self.vercel.app)

- **API Documentation**: [https://rag-based-chatbot-96uz.onrender.com/docs](https://rag-based-chatbot-96uz.onrender.com/docs)
- **Status Page**: [https://stats.uptimerobot.com/5z2EBCHShQ](https://stats.uptimerobot.com/5z2EBCHShQ)
- **Metrics**: [https://rag-based-chatbot-96uz.onrender.com/api/v1/query/metrics](https://rag-based-chatbot-96uz.onrender.com/api/v1/query/metrics)
- **Report**: [https://github.com/jerrybuks/rag-based-chatbot/blob/main/reports/REPORT.MD](https://github.com/jerrybuks/rag-based-chatbot/blob/main/reports/REPORT.MD)

> **Note**: This service is deployed on Render's free tier. The server may be unavailable sometimes due to the free tier limitations. Please check the status page for real-time availability information. If render server is alseep, please give like it 2 minutes and reload 

## API Key Setup

1. You can run this app with either openAI key or [OpenRouter](https://openrouter.ai/) key
   - Sign up for an account at OpenRouter or OpenAI
   - Navigate to API Keys section
   - Create a new API key

2. Add your API key to the `.env` file:
   ```bash
   OPENAI_API_KEY=your_api_key_here
   ```
   if you are using OpenRouter you will also need to add a Base url in your env
    ```bash
   OPENAI_API_BASE=https://openrouter.ai/api/v1
   ```

## Setup

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install .
```

## Running the Application

```bash
fastapi dev src/main.py
```

The server will be available at `http://localhost:8000`

- **API Docs**: http://localhost:8000/docs


## Known Limitations

### Data Persistence
- **File-based Cache & Chroma DB**: 
  - Cache data is stored in memory and file system
  - Chroma DB is stored in memory and file system

### Metrics Storage
- **In-Memory Metrics**: 
  - All metrics are stored in-memory

### Deployment
- Hosted on Render free tier
- Subject to cold starts
- Server sleeps after inactivity
- 2-minute warm-up time may be needed
- Have tried to mitiage this using ```uptimerobot``` that pings my server every 5 minutes, but does not guarnatee 100%