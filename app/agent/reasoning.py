from groq import Groq
import os

def get_client():
    from app.agent.planner import get_client
    return get_client()

def get_async_client():
    from app.agent.planner import get_async_client
    return get_async_client()

async def synthesize_report(query: str, context: list[dict], model: str = "llama-3.1-8b-instant") -> str:
    """
    Generate a structured report with citations based on the query and retrieved context (Async).
    """
    client = get_async_client()
    # Format context for the LLM
    context_str = ""
    for i, item in enumerate(context):
        # Truncate individual chunks if they are huge (eventhough chunking handles this mostly)
        text = item['text'][:1500] 
        context_str += f"[Source {i+1}] ({item['metadata']['source']}):\n{text}\n\n"
    
    # Limit on total context characters to stay well within TPM limits
    if len(context_str) > 6000:
        context_str = context_str[:6000] + "...(truncated)"

    system_prompt = (
        "You are an expert researcher. "
        "Write a structured report based on the provided sources. "
        "Include in-line citations like [Source 1] where appropriate. "
        "Do not invent information not present in the sources. "
        "If sources are insufficient, state that limitations."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Query: {query}\n\nContext:\n{context_str}"}
    ]

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_completion_tokens=1024
    )

    return response.choices[0].message.content

async def generate_chat_response(query: str, model: str = "llama-3.1-8b-instant") -> str:
    """
    Generate a simple conversational response for non-research queries (Async).
    """
    client = get_async_client()
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant. Respond conversationally to the user."},
            {"role": "user", "content": query}
        ],
        temperature=0.7,
        max_completion_tokens=200
    )
    return response.choices[0].message.content
