import httpx
from src.llm.base import BaseProvider, make_retry_decorator
from src.llm.schema import ChatRequest, ChatResponse, UsageStats


class GroqProvider(BaseProvider):

    BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key, **kwargs):
        super().__init__(api_key=api_key, **kwargs)

    def _default_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    async def complete(self, request: ChatRequest) -> ChatResponse:
        retry_deco = make_retry_decorator(max_attempts=self.max_retries)

        @retry_deco
        async def _call():
            client = await self._get_client()

            payload = {
                "model": request.model,
                "messages": [m.to_dict() for m in request.messages],
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
            }

            self._log.debug("Groq request | model=%s messages=%d", request.model, len(request.messages))

            try:
                response = await client.post(self.BASE_URL, json=payload)
            except httpx.TimeoutException:
                self._log.warning("Groq request timed out, will retry...")
                raise
            except httpx.HTTPError as e:
                self._log.warning("Groq network error: %s, will retry...", e)
                raise

            self._raise_for_status("groq", response)

            data = response.json()
            choice = data["choices"][0]["message"]
            usage = data.get("usage", {})

            self._log.info("Groq response | model=%s total_tokens=%s", data.get("model"), usage.get("total_tokens", "?"))

            return ChatResponse(
                content=choice["content"],
                model=data.get("model", request.model),
                provider="groq",
                usage=UsageStats(
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                ) if usage else None,
                raw=data,
            )

        return await _call()