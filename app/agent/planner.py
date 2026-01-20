def get_client():
    """Lazy initialization of the Groq client."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Please ensure the environment variable is configured in your Space or .env file."
        )
    return Groq(api_key=api_key)

def create_plan(query: str) -> list[str]:
    client = get_client()
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
    client = get_client()
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
