from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def create_plan(query: str) -> list[str]:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research planner. "
                    "Break the task into concrete steps. "
                    "Do not execute tools."
                ),
            },
            {"role": "user", "content": query},
        ],
        temperature=0.2,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
    )

    plan_text = response.choices[0].message.content
    return [step.strip("- ") for step in plan_text.splitlines() if step.strip()]

def classify_intent(query: str) -> str:
    """
    Classify the user's intent as either 'RESEARCH' or 'CONVERSATION'.
    """
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an intent classifier. "
                    "Determine if the user's query requires internet research or if it is a simple conversational greeting/question. "
                    "Respond with ONLY the word 'RESEARCH' or 'CONVERSATION'. "
                    "Examples: "
                    "'Hi' -> CONVERSATION "
                    "'Who is the CEO of Google?' -> RESEARCH "
                    "'Explain quantum physics' -> RESEARCH "
                    "'Thank you' -> CONVERSATION"
                ),
            },
            {"role": "user", "content": query},
        ],
        temperature=0.1,
        max_completion_tokens=10,
    )
    return response.choices[0].message.content.strip().upper()
