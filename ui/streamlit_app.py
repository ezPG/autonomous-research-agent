import streamlit as st
import asyncio
import os
import sys
import json
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

st.set_page_config(page_title="Research Agent Chat", layout="wide")

st.title("Auto Research Agent")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "data" in message:
            data = message["data"]
            with st.expander("Research Details"):
                if "plan" in data:
                    st.write("**Plan:**")
                    st.json(data["plan"])
                if "observations" in data:
                    st.write("**Observations:**")
                    st.json(data["observations"])
                if "sources" in data:
                    st.write("**Sources:**")
                    st.json(data["sources"])

# Function to get or create the MCP session
async def get_mcp_session():
    if "mcp_client" not in st.session_state:
        server_params = StdioServerParameters(
            command="python",
            args=["-m", "app.mcp_server"],
            cwd=PROJECT_ROOT,
        )
        # We need to keep the context managers alive
        st.session_state.mcp_cm = stdio_client(server_params)
        read, write = await st.session_state.mcp_cm.__aenter__()
        st.session_state.mcp_session_cm = ClientSession(read, write)
        session = await st.session_state.mcp_session_cm.__aenter__()
        await session.initialize()
        st.session_state.mcp_client = session
    return st.session_state.mcp_client

# Chat input
if prompt := st.chat_input("Ask anything"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🔍 *Agent is planning and searching...*")
        
        try:
            # Using the persistent session
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            session = loop.run_until_complete(get_mcp_session())
            
            result = loop.run_until_complete(session.call_tool(
                "run_research_task",
                arguments={"query": prompt},
            ))
            
            if result and result.content:
                data_str = result.content[0].text
                try:
                    data = json.loads(data_str)
                    report = data.get("report", "No report generated.")
                    message_placeholder.markdown(report)
                    
                    with st.expander("Research Details"):
                        st.write("**Plan:**")
                        st.json(data.get("plan", []))
                        st.write("**Observations:**")
                        st.json(data.get("observations", []))
                        st.write("**Sources:**")
                        st.json(data.get("sources", []))
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": report,
                        "data": data
                    })
                except json.JSONDecodeError as je:
                    error_msg = f"JSON Error: Failed to parse agent response. Raw output: {data_str[:200]}..."
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
            else:
                error_msg = "Error: Agent returned empty result or timed out."
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                
        except Exception as e:
            error_msg = f"An error occurred: {e}"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            # Reset session on error to allow retry with a fresh process
            if "mcp_client" in st.session_state:
                del st.session_state.mcp_client
