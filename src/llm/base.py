import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from abc import ABC, abstractmethod
from src.llm.schema import ChatRequest, ChatResponse
from src.llm.logging_config import get_logger
import logging


def make_retry_decorator(max_attempts=3, min_wait=1.0, max_wait=10.0):
    _retry_logger = logging.getLogger("llm.retry")
    return retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        before_sleep=before_sleep_log(_retry_logger, logging.WARNING),  # logs each retry attempt
        reraise=True,
    )


class ProviderError(Exception):
    def __init__(self, provider: str, status_code: int, message: str):
        self.provider = provider
        self.status_code = status_code
        super().__init__(f"[{provider}] HTTP {status_code}: {message}")


class BaseProvider(ABC):

    def __init__(self, api_key=None, timeout=30.0, max_retries=3):
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = None
        self._log = get_logger(self.__class__.__name__)

    async def _get_client(self):
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers=self._default_headers(),
            )
        return self._client

    def _default_headers(self):
        return {"Content-Type": "application/json"}

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.close()

    @abstractmethod
    async def complete(self, request: ChatRequest) -> ChatResponse:
        ...

    def _raise_for_status(self, provider_name: str, response: httpx.Response):
        if response.status_code >= 400:
            try:
                body = response.json()
                msg = str(body.get("error", body))
            except Exception:
                msg = response.text[:200]
            self._log.error("[%s] HTTP %s: %s", provider_name, response.status_code, msg)
            raise ProviderError(provider_name, response.status_code, msg)