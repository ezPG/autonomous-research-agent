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

st.title("Autonomous Research Agent 🕵️‍♂️")

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

# Function to run the research agent
async def run_research(query):
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "app.mcp_server"],
        cwd=PROJECT_ROOT,
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            result = await session.call_tool(
                "run_research_task",
                arguments={"query": query},
            )
            return result

# Chat input
if prompt := st.chat_input("What would you like me to research today?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🔍 *Agent is planning and searching...*")
        
        try:
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(run_research(prompt))
            
            if result and result.content:
                data_str = result.content[0].text
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
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": report,
                    "data": data
                })
            else:
                error_msg = "Error: Agent returned empty result."
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                
        except Exception as e:
            error_msg = f"An error occurred: {e}"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
