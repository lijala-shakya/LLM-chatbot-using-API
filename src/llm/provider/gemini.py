
import httpx
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
                    "thinkingConfig": {
                        "thinkingBudget": 1024
                    }
                },
            }
            if system_text:
                payload["systemInstruction"] = {"parts": [{"text": system_text}]}

            url = f"{self.BASE_URL}/{request.model}:generateContent?key={self.api_key}"

            self._log.debug("Gemini request | model=%s messages=%d", request.model, len(request.messages))

            try:
                response = await client.post(url, json=payload)
            except httpx.TimeoutException:
                self._log.warning("Gemini request timed out, will retry...")
                raise
            except httpx.HTTPError as e:
                self._log.warning("Gemini network error: %s, will retry...", e)
                raise

            self._raise_for_status("gemini", response)

            data = response.json()
            candidate = data["candidates"][0]
            content_text = candidate["content"]["parts"]
            # content_text = candidate["content"]["parts"][0]["text"]
            thought_text = ""
            answer_text = ""

            for part in candidate["content"]["parts"]:
                if part.get("thought", False):
                    thought_text = part.get("text", "")
                else:
                    answer_text = part.get("text", "")

            if thought_text and request.extra.get("thinking", False):
                content_text = f"[Thinking]\n{thought_text}\n\n[Answer]\n{answer_text}"
            else:
                content_text = answer_text
            usage_meta = data.get("usageMetadata", {})

            self._log.info("Gemini response | model=%s total_tokens=%s", request.model, usage_meta.get("totalTokenCount", "?"))

            return ChatResponse(
                content=content_text,
                model=request.model,
                provider="gemini",
                usage=UsageStats(
                    prompt_tokens=usage_meta.get("promptTokenCount", 0),
                    completion_tokens=usage_meta.get("candidatesTokenCount", 0),
                    total_tokens=usage_meta.get("totalTokenCount", 0),
                ) if usage_meta else None,
                raw=data,
            )

        return await _call()