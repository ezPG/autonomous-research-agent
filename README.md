
---
title: Autonomous Research Agent
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Autonomous Research Agent (RAG + MCP)

An end-to-end AI research agent that plans research tasks, searches the web, fetches documents, builds a RAG index, and generates structured, citation-backed reports. Built with **FastMCP**, **LangChain**, and **Groq**.

**[Live Demo on Azure Container Apps](https://research-agent-app.calmcoast-5e3b9fd4.centralus.azurecontainerapps.io/)** | **[Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/ezPG/auto-research-agent)**


## Features
- **Autonomous Reasoning (ReAct)**: A dynamic reasoning loop (Plan-Observe-Execute) that thrashes out complex queries autonomously.
- **Local PDF Ingestion**: Upload and index local PDFs directly into the RAG store via the sidebar.
- **Persistent RAG**: Uses FAISS and SentenceTransformers with disk persistence for reliable knowledge retrieval.
- **Native MCP Architecture**: Decoupled agent (client) and tools (server) architecture using the Model Context Protocol.
- **Modern UI**: Streamlit-based chat interface with real-time reasoning logs.

## Getting Started

### 1. Prerequisite: API Key
You only need a **Groq API Key** to get started. Obtain one at the [Groq Console](https://console.groq.com/).

### 2. Run with Docker (Recommended)
The easiest way to run the agent is using Docker, which handles all dependencies.

```bash
# Clone the repo
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
