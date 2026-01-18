from groq import Groq
import os
from dotenv import load_dotenv
from app.rag import RAGStore
from app.mcp_context import MCPContext
from app.rag import RAGStore



load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
rag_store = RAGStore()

def ingest_document(text: str):
    rag_store.add_document(text)
def generate_answer(prompt: str) -> str:
    
    retrieved = rag_store.retrieve(prompt)
    
    mcp_context = MCPContext(
        query=prompt,
        retrieved_chunks=retrieved,
        tools_used=["faiss_retrieval"],
        memory={}
    )
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a research assistant."},
            {"role": "user", "content": mcp_context.to_prompt()}
        ],
        temperature=0.2,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
        # stop=None
    )
    return response.choices[0].message.content
