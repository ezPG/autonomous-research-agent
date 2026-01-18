from dataclasses import dataclass
from typing import List, Dict

@dataclass
class MCPContext:
    query: str
    retrieved_chunks: List[str]
    tools_used: List[str]
    memory: Dict

    def to_prompt(self) -> str:
        context = "Retrieved context:\n"
        for chunk in self.retrieved_chunks:
            context += f"- {chunk}\n"

        return f"""
You are an autonomous research agent.

User Query:
{self.query}

{context}

Answer using ONLY the retrieved context.
Cite facts implicitly.
"""
