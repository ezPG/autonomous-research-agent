from fastapi import FastAPI
from pydantic import BaseModel
from app.agent import generate_answer, ingest_document
from app.tools.ingest import load_pdf

app = FastAPI(title="Autonomous Research Agent")

class QueryRequest(BaseModel):
    query: str
    
class IngestRequest(BaseModel):
    path: str

@app.post("/ingest")
def ingest(req: IngestRequest):
    text = load_pdf(req.path)
    ingest_document(text)
    return {"status": "document ingested"}

@app.post("/query")
def query_agent(req: QueryRequest):
    answer = generate_answer(req.query)
    return {
        "query": req.query,
        "answer": answer
    }
