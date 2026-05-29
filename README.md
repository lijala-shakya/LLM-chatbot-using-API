# LLM Chat 

A modular async Python chatbot supporting Gemini and Groq with conversation memory, logging, and retry mechanism.

---

## Features

- Multi-provider support — Gemini and Groq
- Conversation memory — full history sent each turn
- Model selection at startup
- Switch model mid-conversation with `/switch`
- Retry logic with exponential backoff
- Structured logging to file
- Pydantic validation on all requests and responses
- Graceful error handling


## Project Structure


project/
├── main.py                  # Entry point and chat loop
├── .env                     # Your API keys (never commit this)
├── .env.example             # Template for API keys
├── .gitignore
├── pyproject.toml           # Dependencies managed by UV
├── uv.lock                  # Locked dependency versions
├── logs/
│   └── chat.log             # Auto-created at runtime
└── src/
    └── llm/
        ├── schema.py        # Pydantic models
        ├── base.py          # BaseProvider and retry logic
        ├── client.py        # AIClient
        ├── logging_config.py
        └── provider/
            ├── __init__.py
            ├── gemini.py
            └── groq.py


## Requirements

- Python 3.12+
- UV package manager
- Gemini API key (free) — aistudio.google.com
- Groq API key (free) — console.groq.com


## Setup

**1. Install UV** (one time only)

Windows:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Mac/Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Clone the repository**
```bash
git clone https://github.com/your-username/llm-chat-cli
cd llm-chat-cli
```

**3. Install dependencies**
```bash
uv sync
```

**4. Set up environment variables**
```bash
cp .env.example .env
```

Open `.env` and add your API keys:
```
GROQ_API_KEY=your_groq_key_here
GEMINI_API_KEY=your_gemini_key_here
```

**5. Run**
```bash
uv run python main.py
```


## Usage

When you start the app you will be asked to select a model:

```
=== Select a model ===
  1. Gemini 2.5 Flash
  2. Groq Llama 3.3 70b

Enter number (1-2):
```

Then just start chatting:

```
Chatting with gemini / gemini-2.5-flash
Type 'exit' to quit


## Commands

| Command | Description |
|---|---|
| `/switch` | Switch to a different model |
| `exit` / `quit` / `bye` | End the session |

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes (if using Gemini) | Google AI Studio API key |
| `GROQ_API_KEY` | Yes (if using Groq) | Groq Console API key |

At least one key must be set.


## Models

| Option | Provider | Model |
|---|---|---|
| 1 | Gemini | gemini-2.5-flash |
| 2 | Groq | llama-3.3-70b-versatile |


## How it works

**Memory** — the full conversation history is sent with every request so the model understands follow-up questions like "make it shorter" or "give me an example" without needing extra context.

**Retry** — if a request fails due to a network error or timeout, it automatically retries up to 3 times with exponential backoff (waits 1s, then 2s, then 4s between attempts).

**Logging** — every request, response, and error is logged to `logs/chat.log` with timestamps. The terminal only shows warnings and errors so logs don't interrupt the chat.

**Error handling** — if an API call fails, the error is shown in the terminal, the failed message is removed from history, and you can try again without restarting.


## Getting API Keys

**Gemini (free)**
1. Go to aistudio.google.com
2. Sign in with Google
3. Click Get API Key

**Groq (free)**
1. Go to console.groq.com
2. Sign up
3. Go to API Keys → Create API Key


## .env.example

```
GROQ_API_KEY=your_groq_key_here
GEMINI_API_KEY=your_gemini_key_here
```

---

## .gitignore

```
.env
.venv/
logs/
__pycache__/
*.pyc
```

***Setup Instructions***

&#x20;*Clone the Repository*

```bash

*git clone https:https:**//github.com/lijala-shakya/LLM-chatbot-using-API***

**cd llm-chatbot**