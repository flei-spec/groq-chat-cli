import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Read API key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "Missing GROQ_API_KEY. Create a .env file with GROQ_API_KEY=... or export it in your terminal."
    )

# Groq OpenAI-compatible endpoint
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Model: use the same one you used in your screenshot
MODEL = "openai/gpt-oss-120b"


def call_groq(user_question: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_question},
        ],
        "temperature": 0.7,
        "max_completion_tokens": 512,
        "stream": False,
    }

    response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]



def main():
    print("Groq Chat CLI")
    print("Type your question and press Enter.")
    print("Type 'quit' to exit.\n")

    while True:
        user_text = input("You: ").strip()

        if user_text.lower() == "quit":
            print("Exiting. Bye!")
            break

        if user_text == "":
            # skip empty lines
            continue

        try:
            answer = call_groq(user_text)
            print("\nAssistant:", answer, "\n")
        except requests.HTTPError as e:
            print("\nHTTP error:", e)
        except Exception as e:
            print("\nError:", e)


if __name__ == "__main__":
    main()