from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class AgentState:
    query: str
    plan: List[str] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    sources: List[Dict] = field(default_factory=list)
    memory: Dict = field(default_factory=dict)
    tools_used: List[str] = field(default_factory=list)