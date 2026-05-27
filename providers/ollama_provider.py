import time
from typing import Iterator, Optional

from ollama import Client, ResponseError

from .base import ChatProvider, ProviderError


class OllamaProvider(ChatProvider):
    """本地 Ollama 模型。num_ctx / num_predict 每次从 settings 动态读取。"""

    name = "ollama"

    def __init__(self, model: str = "qwen2.5:7b", base_url: Optional[str] = None,
                 timeout: int = 300, max_retries: int = 5, retry_wait: int = 3):
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_wait = retry_wait
        self._client = Client(host=base_url, timeout=timeout) if base_url else Client(timeout=timeout)

    def stream_chat(self, messages: list[dict]) -> Iterator[str]:
        from settings import load_settings
        s = load_settings()
        options = {
            "num_ctx":     s["num_ctx"],     # settings.py → num_ctx
            "num_predict": s["num_predict"],  # settings.py → num_predict
        }
        for attempt in range(self.max_retries):
            try:
                stream = self._client.chat(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    keep_alive=-1,
                    options=options,
                )
                for part in stream:
                    content = part["message"]["content"]
                    if content:
                        yield content
                return
            except ResponseError as e:
                if getattr(e, "status_code", None) == 502 and attempt < self.max_retries - 1:
                    time.sleep(self.retry_wait)
                    continue
                raise ProviderError(
                    f"Ollama 调用失败：{e}",
                    retryable=(getattr(e, "status_code", None) == 502),
                )
            except Exception as e:
                raise ProviderError(f"Ollama 连接错误：{e}")
