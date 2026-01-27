import json
from typing import List, Dict, Any, Optional
from mcp.client.session import ClientSession
from app.agent.state import AgentState
from app.agent.planner import get_client
from app.agent.reasoning import synthesize_report

class ResearchAgent:
    """
    A dynamic research agent that uses a reasoning loop (Plan-Observe-Execute)
    to perform autonomous research via MCP tools.
    """

    def __init__(self, session: ClientSession, model: str = "llama-3.1-8b-instant"):
        from app.agent.planner import get_async_client
        self.session = session
        self.model = model
        self.client = get_async_client()

    async def run(self, query: str, max_iterations: int = 5) -> AgentState:
        from app.agent.planner import classify_intent
        from app.agent.reasoning import generate_chat_response

        state = AgentState(query=query)
        
        # Classify Intent
        try:
            intent = await classify_intent(query, model=self.model)
        except Exception as e:
            state.observations.append(f"Intent classification failed: {e}")
            intent = "RESEARCH" # Default to research if classification fails
        if intent == "CONVERSATION":
            state.observations.append("Classified as conversational query.")
            try:
                state.report = await generate_chat_response(query, model=self.model)
            except Exception as e:
                state.report = f"Failed to generate chat response: {e}"
            return state

        # Initial Planning
        system_prompt = (
            "You are an autonomous research agent. Your goal is to research the user's query thoroughly. "
            "You have access to tools via MCP. Use them to gather information, index it into RAG, and eventually synthesize a report. "
            "Users may have already indexed local documents (like PDFs) into the RAG store; use `query_rag` to check for this existing knowledge first.\n\n"
            "You should work in steps: Reasoning -> Action -> Observation.\n\n"
            "Available tools:\n"
            "- web_search(query: str): Returns a list of URLs.\n"
            "- fetch_page_content(url: str): Returns text content of a page/blog and indexes it.\n"
            "- fetch_pdf_content(url: str): Returns text content of a PDF URL and indexes it.\n"
            "- query_rag(query: str): Returns relevant snippets from indexed content (including local uploads).\n\n"
            "Format your response as a JSON object with two fields:\n"
            "1. 'thought': Your reasoning about what to do next.\n"
            "2. 'action': The tool call to make, e.g., {'name': 'web_search', 'arguments': {'query': '...'}} or {'name': 'complete', 'arguments': {}} when done. "
            "You MUST use tools to gather information for research queries. Do NOT rely on your internal training data to answer. "
            "Only finish when you have gathered enough information and indexed it."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}"}
        ]

        for i in range(max_iterations):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                decision = json.loads(response.choices[0].message.content)
            except Exception as e:
                err_msg = f"Iteration {i+1} failed at LLM reasoning: {e}"
                state.observations.append(err_msg)
                state.report = f"Research loop aborted due to LLM error: {e}"
                break
            
            thought = decision.get("thought", "")
            action = decision.get("action", {})
            
            state.observations.append(f"Thought: {thought}")
            messages.append({"role": "assistant", "content": response.choices[0].message.content})

            action_name = action.get("name")
            action_args = action.get("arguments", {})

            if action_name == "complete" or not action_name:
                state.observations.append("Research complete.")
                break

            state.tools_used.append(action_name)
            state.observations.append(f"Action: {action_name}({action_args})")

            # Execute tool via MCP
            observation = "No input from tool."
            try:
                result = await self.session.call_tool(action_name, arguments=action_args)
                if result and result.content:
                    observation = result.content[0].text
                
                state.observations.append(f"Observation: {observation[:200]}...")
                
                # Safety Truncation for history:
                # Agent context could explode. If too long (raw dump), truncate.
                trunc_obs = observation if len(observation) < 2000 else observation[:2000] + "...(truncated)"
                messages.append({"role": "user", "content": f"Observation: {trunc_obs}"})
                
            except Exception as e:
                error_msg = f"Error executing tool {action_cmd}: {e}"
                state.observations.append(f"Observation: {error_msg}")
                messages.append({"role": "user", "content": error_msg})

        # Final Synthesis
        if state.report:
            return state

        # Check RAG for collected info
        try:
            rag_results = await self.session.call_tool("query_rag", arguments={"query": query, "k": 5})
            if rag_results and rag_results.content:
                # The synthesize_report expects a list of dicts with 'text' and 'metadata'
                context = json.loads(rag_results.content[0].text)
                state.report = await synthesize_report(query, context, model=self.model)
            else:
                state.report = "No information gathered to synthesize a report."
        except Exception as e:
            state.report = f"Failed to synthesize report: {e}"

        return state
