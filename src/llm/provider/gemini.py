import httpx
import json
from src.llm.base import BaseProvider, make_retry_decorator
from src.llm.schema import ChatRequest, ChatResponse, UsageStats, Message


def _messages_to_gemini(messages: list[Message]):
    system_instruction = None
    contents = []

    for msg in messages:
        if msg.role == "system":
            system_instruction = msg.content
        elif msg.role == "user":
            contents.append({"role": "user", "parts": [{"text": msg.content}]})
        elif msg.role == "assistant":
            contents.append({"role": "model", "parts": [{"text": msg.content}]})

    return system_instruction, contents


class GeminiProvider(BaseProvider):

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key, **kwargs):
        super().__init__(api_key=api_key, **kwargs)

    def _default_headers(self):
        return {"Content-Type": "application/json"}

    async def complete(self, request: ChatRequest) -> ChatResponse:
        retry_deco = make_retry_decorator(max_attempts=self.max_retries)

        @retry_deco
        async def _call():
            client = await self._get_client()
            system_text, contents = _messages_to_gemini(request.messages)

            payload = {
                "contents": contents,
                "generationConfig": {
                    "maxOutputTokens": request.max_tokens,
                    "temperature": request.temperature,
                },
            }
            if system_text:
                payload["systemInstruction"] = {"parts": [{"text": system_text}]}

            # CHANGED: streaming endpoint instead of generateContent
            url = f"{self.BASE_URL}/{request.model}:streamGenerateContent?key={self.api_key}&alt=sse"

            self._log.debug("Gemini request | model=%s messages=%d", request.model, len(request.messages))

            full_text = ""
            usage_meta = {}

            try:
                # CHANGED: client.stream() instead of client.post()
                async with client.stream("POST", url, json=payload) as response:
                    print(f"DEBUG status: {response.status_code}")
                    if response.status_code >= 400:
                        await response.aread()
                        self._raise_for_status("gemini", response)

                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[len("data:"):].strip()
                        if not raw:
                            continue
                        try:
                            chunk = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if "error" in chunk:
                            error = chunk["error"]
                            raise Exception(f"Gemini error {error.get('code')}: {error.get('message')}")

                        candidates = chunk.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            for part in parts:
                                text = part.get("text", "")
                                if text:
                                    print(text, end="", flush=True)
                                    full_text += text

                        if "usageMetadata" in chunk:
                            usage_meta = chunk["usageMetadata"]

            except httpx.TimeoutException:
                self._log.warning("Gemini request timed out, will retry...")
                raise
            except httpx.HTTPError as e:
                self._log.warning("Gemini network error: %s, will retry...", e)
                raise

            self._log.info("Gemini response | model=%s total_tokens=%s", request.model, usage_meta.get("totalTokenCount", "?"))

            return ChatResponse(
                content=full_text,
                model=request.model,
                provider="gemini",
                usage=UsageStats(
                    prompt_tokens=usage_meta.get("promptTokenCount", 0),
                    completion_tokens=usage_meta.get("candidatesTokenCount", 0),
                    total_tokens=usage_meta.get("totalTokenCount", 0),
                ) if usage_meta else None,
                raw={},
            )

        return await _call()