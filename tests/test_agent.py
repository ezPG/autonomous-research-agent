import pytest
from unittest.mock import MagicMock, patch
from app.rag import RAGStore
from app.tools.fetcher import fetch_url
from app.tools.web_search import duckduckgo_search

# Test RAG functionality
def test_rag_store_indexing():
    store = RAGStore()
    store.add_document("Artificial intelligence is transforming the world.", source="ai_doc")
    store.add_document("Quantum computing is the next frontier.", source="qc_doc")
    
    # Verify both documents are in metadata
    assert len(store.text_chunks) >= 2
    assert any(m["source"] == "ai_doc" for m in store.metadata)
    assert any(m["source"] == "qc_doc" for m in store.metadata)

def test_rag_retrieval_accuracy():
    store = RAGStore()
    store.add_document("The quick brown fox jumps over the lazy dog.", source="animal")
    store.add_document("The chef prepared a delicious spicy pasta for dinner.", source="food")
    
    results = store.retrieve("What did the cook make?", k=1)
    assert len(results) == 1
    assert "chef" in results[0]["text"].lower()
    assert results[0]["metadata"]["source"] == "food"

def test_rag_empty_state():
    store = RAGStore()
    assert store.retrieve("anything") == []
    # Test adding empty text
    store.add_document("")
    assert len(store.text_chunks) == 0

# Mocking LangChain Tools
@patch('app.tools.web_search.DuckDuckGoSearchAPIWrapper')
def test_mock_web_search(mock_wrapper):
    # Setup mock return value
    mock_instance = mock_wrapper.return_value
    mock_instance.results.return_value = [
        {"link": "https://example.com/1", "title": "Result 1"},
        {"link": "https://example.com/2", "title": "Result 2"},
    ]
    
    links = duckduckgo_search("test query", max_results=2)
    
    assert len(links) == 2
    assert links[0] == "https://example.com/1"
    mock_instance.results.assert_called_once()

@patch('app.tools.fetcher.WebBaseLoader')
def test_mock_fetch_url(mock_loader):
    # Setup mock document
    mock_doc = MagicMock()
    mock_doc.page_content = "This is a mocked webpage content."
    mock_loader.return_value.load.return_value = [mock_doc]
    
    content = fetch_url("https://example.com")
    assert "mocked webpage" in content
    mock_loader.assert_called_once_with("https://example.com")

# Test for the planner utility (Mocks Groq API)
@patch('app.agent.planner.client')
def test_planner_logic(mock_groq_client):
    # Mock recursive structure of Groq response
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "- Step 1\n- Step 2\n- Step 3"
    mock_groq_client.chat.completions.create.return_value = mock_response
    
    from app.agent.planner import create_plan
    plan = create_plan("How to build a car?")
    
    assert len(plan) == 3
    assert plan[0] == "Step 1"
    mock_groq_client.chat.completions.create.assert_called_once()
