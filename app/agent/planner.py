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
