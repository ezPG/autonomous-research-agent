from groq import Groq
import os

# Handle missing API key gracefully for CI/CD environments
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    # Use a dummy key if not present (tests will mock the client anyway)
    api_key = "gsk_dummy_key_for_ci_environments"

client = Groq(api_key=api_key)

def synthesize_report(query: str, context: list[dict]) -> str:
    """
    Generate a structured report with citations based on the query and retrieved context.
    """
    # Format context for the LLM
    context_str = ""
    for i, item in enumerate(context):
        # Truncate individual chunks if they are huge (though chunking handles this mostly)
        text = item['text'][:1500] 
        context_str += f"[Source {i+1}] ({item['metadata']['source']}):\n{text}\n\n"
    
    # Hard limit on total context characters to stay well within TPM limits
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

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.3,
        max_completion_tokens=1024
    )

    return response.choices[0].message.content
