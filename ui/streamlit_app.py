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

# Sidebar for PDF Ingestion
with st.sidebar:
    st.header("Upload Documents")
    uploaded_file = st.file_uploader("Index a local PDF", type=["pdf"])
    
    st.header("Settings")
    model_choice = st.selectbox(
        "Select LLM Model",
        ["llama-3.1-8b-instant", "openai/gpt-oss-20b"],
        index=0
    )
    
    if uploaded_file and "indexed_files" not in st.session_state:
        st.session_state.indexed_files = set()

    if uploaded_file and uploaded_file.name not in st.session_state.get("indexed_files", set()):
        with st.status(f"Indexing {uploaded_file.name}...") as status:
            import tempfile
            from langchain_community.document_loaders import PyMuPDFLoader
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            try:
                loader = PyMuPDFLoader(tmp_path)
                docs = loader.load()
                full_text = "\n".join([doc.page_content for doc in docs])
                
                if not full_text.strip():
                    st.warning(f"No text extracted from {uploaded_file.name}. It might be a scanned PDF or empty.")
                else:
                    # Start an active MCP session to index. 
                    async def index_local_text(text, source):
                        env = os.environ.copy()
                        env["PYTHONPATH"] = PROJECT_ROOT
                        server_params = StdioServerParameters(
                            command=sys.executable,
                            args=["-m", "app.mcp_server"],
                            cwd=PROJECT_ROOT,
                            env=env
                        )
                        async with stdio_client(server_params) as (read, write):
                            async with ClientSession(read, write) as session:
                                await session.initialize()
                                await session.call_tool("index_text", arguments={"text": text, "source": source})
                    
                    try:
                        import nest_asyncio
                        nest_asyncio.apply()
                        asyncio.run(index_local_text(full_text, uploaded_file.name))
                    except Exception as loop_error:
                        # Fallback for complex loop envs
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(index_local_text(full_text, uploaded_file.name))
                    
                    st.session_state.indexed_files.add(uploaded_file.name)
                    status.update(label=f"Finished indexing {uploaded_file.name}", state="complete")
                    st.success(f"Indexed {uploaded_file.name}")
            except Exception as e:
                import traceback
                st.error(f"Failed to index PDF: {e}")
                st.code(traceback.format_exc())
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    if st.session_state.get("indexed_files"):
        st.write("**Indexed Files:**")
        for f in st.session_state.indexed_files:
            st.write(f"- {f}")
        
        if st.button("Clear RAG Index"):
            async def clear_remote_rag():
                env = os.environ.copy()
                env["PYTHONPATH"] = PROJECT_ROOT
                server_params = StdioServerParameters(
                    command=sys.executable,
                    args=["-m", "app.mcp_server"],
                    cwd=PROJECT_ROOT,
                    env=env
                )
                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        await session.call_tool("clear_rag", arguments={})
            
            try:
                import nest_asyncio
                nest_asyncio.apply()
                asyncio.run(clear_remote_rag())
                st.session_state.indexed_files = set()
                st.success("RAG Index cleared.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to clear RAG: {e}")

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
                if "observations" in data:
                    st.write("**Observations:**")
                    st.json(data["observations"])
                if "sources" in data:
                    st.write("**Sources:**")
                    st.json(data["sources"])

# Function to run the research agent
async def run_research(query, log_placeholder, model):
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp_server"],
        cwd=PROJECT_ROOT,
        env=env
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            from app.agent.agent import ResearchAgent
            agent = ResearchAgent(session, model=model)
            
            log_placeholder.write("🚀 Agent started research loop...")
            
            state = await agent.run(query)
            return state

# Chat input
if prompt := st.chat_input("Ask anything"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        log_placeholder = st.expander("Agent Reasoning Logs", expanded=True)
        
        try:
            state = asyncio.run(run_research(prompt, log_placeholder, model_choice))
            
            if state:
                report = state.report
                message_placeholder.markdown(report)
                
                # Show observations in logs
                with log_placeholder:
                    for obs in state.observations:
                        if obs.startswith("Thought:"):
                            st.write(f"{obs}")
                        elif obs.startswith("Action:"):
                            st.write(f"{obs}")
                        elif obs.startswith("Observation:"):
                            st.write(f"{obs}")
                        else:
                            st.write(obs)

                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": report,
                    "data": {
                        "observations": state.observations,
                        "sources": state.sources
                    }
                })
            else:
                error_msg = "Error: Agent failed to return state."
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                
        except Exception as e:
            import traceback
            inner_errors = []
            if isinstance(e, BaseExceptionGroup):
                for exc in e.exceptions:
                    inner_errors.append(str(exc))
            else:
                inner_errors.append(str(e))
                
            error_details = "\n".join([f"- {err}" for err in inner_errors])
            error_msg = f"An error occurred during research:\n{error_details}"
            
            message_placeholder.error(error_msg)
            with st.expander("Full Traceback"):
                st.code(traceback.format_exc())
                
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
