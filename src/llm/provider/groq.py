import httpx
import json
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
                "stream": True,  # CHANGED: enable streaming
            }

            self._log.debug("Groq request | model=%s messages=%d", request.model, len(request.messages))

            full_text = ""
            usage = {}

            try:
                # CHANGED: client.stream() instead of client.post()
                async with client.stream("POST", self.BASE_URL, json=payload) as response:
                    self._raise_for_status("groq", response)

                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[len("data:"):].strip()
                        if raw == "[DONE]":
                            break
                        try:
                            chunk = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        choices = chunk.get("choices", [])
                        if choices:
                            text = choices[0].get("delta", {}).get("content", "")
                            if text:
                                print(text, end="", flush=True)
                                full_text += text

                        if chunk.get("usage"):
                            usage = chunk["usage"]

            except httpx.TimeoutException:
                self._log.warning("Groq request timed out, will retry...")
                raise
            except httpx.HTTPError as e:
                self._log.warning("Groq network error: %s, will retry...", e)
                raise

            self._log.info("Groq response | model=%s total_tokens=%s", request.model, usage.get("total_tokens", "?"))

            return ChatResponse(
                content=full_text,
                model=request.model,
                provider="groq",
                usage=UsageStats(
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                ) if usage else None,
                raw={},
            )

        return await _call()