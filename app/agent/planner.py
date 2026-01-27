from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

def get_client():
    """Lazy initialization of the Groq client (Sync)."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set.")
    return Groq(api_key=api_key)

def get_async_client():
    """Lazy initialization of the AsyncGroq client."""
    from groq import AsyncGroq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set.")
    return AsyncGroq(api_key=api_key)


async def classify_intent(query: str, model: str = "llama-3.1-8b-instant") -> str:
    """
    Classify the user's intent as either 'RESEARCH' or 'CONVERSATION' (Async).
    """
    client = get_async_client()
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an intent classifier. "
                    "Determine if the user's query requires internet research or if it is a simple conversational greeting, localized question, or polite closing. "
                    "One-word greetings (Hi, Hello, Hey) and polite statements (Thanks, Bye) should be CONVERSATION. "
                    "Respond with ONLY the word 'RESEARCH' or 'CONVERSATION'. "
                    "Examples: "
                    "'Hi' -> CONVERSATION "
                    "'Hello there' -> CONVERSATION "
                    "'Who is the CEO of Google?' -> RESEARCH "
                    "'Explain quantum physics' -> RESEARCH "
                    "'Thank you for your help' -> CONVERSATION"
                ),
            },
            {"role": "user", "content": query},
        ],
        temperature=0.1,
        max_completion_tokens=10,
    )
    return response.choices[0].message.content.strip().upper()
