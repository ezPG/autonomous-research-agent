from fastapi import FastAPI
from pydantic import BaseModel
from app.agent import generate_answer

app = FastAPI(title="Autonomous Research Agent")

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def query_agent(req: QueryRequest):
    answer = generate_answer(req.query)
    return {
        "query": req.query,
        "answer": answer
    }
