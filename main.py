import asyncio
import os
from dotenv import load_dotenv
from src.llm.client import AIClient
from src.llm.schema import Message
from src.llm.logging_config import setup_logging

load_dotenv()
setup_logging(log_level="INFO", log_file="logs/chat.log")

PROVIDERS = {
    "1": {"provider": "gemini", "model": "gemini-2.5-flash",        "label": "Gemini 2.5 Flash"},
    "2": {"provider": "groq",   "model": "llama-3.3-70b-versatile", "label": "Groq Llama 3.3 70b"},
}

def select_model() -> tuple[str, str]:
    print("\n=== Select a model ===")
    for key, value in PROVIDERS.items():
        print(f"  {key}. {value['label']}")

    while True:
        choice = input("\nEnter number (1-2): ").strip()
        if choice in PROVIDERS:
            selected = PROVIDERS[choice]
            print(f"\nUsing: {selected['label']}\n")
            return selected["provider"], selected["model"]
        print("Invalid choice, enter 1 or 2.")

async def main():
    async with AIClient(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
    ) as client:

        provider, model = select_model()
        thinking = False

        history = [
            Message(
                role="system",
                content="You are a helpful assistant. When the user asks multiple questions, answer every single one of them clearly and completely. Never skip a question.",
            )
        ]

        print(f"Chatting with {provider} / {model}")
        print("Type 'exit' to quit | '/switch' to change model | '/think' to toggle thinking\n")

        while True:
            user_input = input("you: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "bye"):
                print("Goodbye!")
                break

            if user_input.lower() == "/switch":
                provider, model = select_model()
                print(f"Switched to {provider} / {model}\n")
                continue

            if user_input.lower() == "/think":
                thinking = not thinking
                status = "ON" if thinking else "OFF"
                print(f"Thinking mode: {status}\n")
                continue

            history.append(Message(role="user", content=user_input))

            try:
                print("\nAI: ", end="", flush=True)  # print AI: before streaming starts

                response = await client.chat(
                    provider=provider,
                    model=model,
                    messages=history,
                    max_tokens=4096,
                    thinking=thinking,
                )

                print("\n")       
                history.append(Message(role="assistant", content=response.content))

            except Exception as e:
                print(f"\nError: {e}\n")
                history.pop()


asyncio.run(main())