import asyncio
from dotenv import load_dotenv
from src.llm.schema import ChatRequest, ChatResponse, Message
from src.llm.provider.groq import GroqProvider
from src.llm.provider.gemini import GeminiProvider
from src.llm.logging_config import get_logger

load_dotenv()

log = get_logger("client")


class AIClient:

    def __init__(
        self,
        groq_api_key=None,
        gemini_api_key=None,
        timeout=30.0,
        max_retries=3,
    ):
        shared_kwargs = {"timeout": timeout, "max_retries": max_retries}
        self._providers = {}

        if groq_api_key:
            self._providers["groq"] = GroqProvider(groq_api_key, **shared_kwargs)
            log.info("Registered provider: groq")

        if gemini_api_key:
            self._providers["gemini"] = GeminiProvider(gemini_api_key, **shared_kwargs)
            log.info("Registered provider: gemini")

        if not self._providers:
            raise ValueError("No API keys provided — set at least one in your .env file.")

    async def chat(
        self,
        provider: str,
        model: str,
        messages: list[Message],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        thinking: bool = False,  # ADD 1: new parameter
    ) -> ChatResponse:
        if provider not in self._providers:
            available = list(self._providers.keys())
            raise ValueError(f"Provider '{provider}' not configured. Available: {available}")

        request = ChatRequest(
            messages=messages,
            model=model,
            provider=provider,
            max_tokens=max_tokens,
            temperature=temperature,
            extra={"thinking": thinking},  # ADD 2: pass thinking through extra
        )

        log.info("chat() | provider=%s model=%s thinking=%s", provider, model, thinking)
        response = await self._providers[provider].complete(request)
        log.info("chat() done | tokens=%s", response.usage.total_tokens if response.usage else "?")
        return response

    async def chat_all(self, model_map, messages, **kwargs):
        tasks = {
            provider: self.chat(provider, model, messages, **kwargs)
            for provider, model in model_map.items()
            if provider in self._providers
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        return dict(zip(tasks.keys(), results))

    async def close(self):
        for provider in self._providers.values():
            await provider.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.close()