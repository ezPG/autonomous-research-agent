from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

# Handle missing API key gracefully for CI/CD environments
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    # Use a dummy key if not present (tests will mock the client anyway)
    api_key = "gsk_dummy_key_for_ci_environments"

client = Groq(api_key=api_key)

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
