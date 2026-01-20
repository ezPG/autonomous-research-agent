
# Autonomous Research Agent (RAG + MCP)

An end-to-end AI research agent that plans research tasks, searches the web, fetches documents, builds a RAG index, and generates structured, citation-backed reports. Built with **FastMCP**, **LangChain**, and **Groq**.


## Features
- **Autonomous Planning**: Decomposes complex queries into actionable research steps.
- **Multi-Source Fetching**: Scrapes web pages and PDFs using LangChain Community tools.
- **RAG Architecture**: In-memory FAISS index with SentenceTransformer embeddings for precise retrieval.
- **MCP Native**: Exposes tools (`web_search`, `fetch_url`, `query_rag`) and resources (`rag://knowledge`) via the Model Context Protocol.
- **Modern UI**: Streamlit-based chat interface for an interactive research experience.
- **CLI Mode**: Direct terminal entrypoint for quick research tasks.

## Tech Stack
- **LLM**: Groq (Llama-3.1-8b-instant)
- **Framework**: LangChain (Community tools)
- **Agent Protocol**: FastMCP (Model Context Protocol)
- **Vector Store**: FAISS
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **Frontend**: Streamlit

## Getting Started

### 1. Prerequisite: API Key
You only need a **Groq API Key** to get started. Obtain one at the [Groq Console](https://console.groq.com/).

### 2. Run with Docker (Recommended)
The easiest way to run the agent is using Docker, which handles all dependencies.

```bash
# Clone the repository
git clone https://github.com/ezPG/autonomous-research-agent
cd autonomous-research-agent

# Build and run the container
docker build -t research-agent .
docker run -p 8501:8501 -e GROQ_API_KEY=[your_api_key] research-agent
```
Access the chat UI at `http://localhost:8501`.

### 3. Manual Installation
```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_api_key_here
streamlit run ui/streamlit_app.py
```

## Usage

### CLI Mode
```bash
PYTHONPATH=. python -m app.mcp_server "How does quantum entanglement work?"
```

## Testing
```bash
python -m pytest
```

## Project Structure
```text
.
├── app/
│   ├── agent/           # Reasoning, Planning, and State logic
│   ├── tools/           # Search and Fetching tools (LangChain)
│   ├── mcp_server.py    # Main FastMCP Server & Orchestration
│   └── rag.py           # FAISS Vector Store management
├── ui/
│   └── streamlit_app.py # Chat-based Frontend
├── tests/               # Unit tests with mocks
├── Dockerfile           # Containerization setup
└── requirements.txt     # Dependencies
```
