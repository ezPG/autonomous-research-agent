from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_answer(prompt: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a research assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        # stop=None
    )
    return response.choices[0].message.content
