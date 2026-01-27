from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class AgentState:
    query: str
    observations: List[str] = field(default_factory=list)
    sources: List[Dict] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    report: str = ""