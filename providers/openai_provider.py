from typing import Iterator

from openai import OpenAI, APIError, APIConnectionError, AuthenticationError, RateLimitError

from .base import ChatProvider, ProviderError


class OpenAIProvider(ChatProvider):
    """OpenAI 协议兼容的所有厂商：OpenAI / DeepSeek / 智谱 / Kimi / 通义 / OpenRouter / vLLM 等。

    只需要换 base_url 和 api_key、model 即可。
    """

    name = "openai"

    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1",
                 api_key: str = "", timeout: int = 120):
        if not api_key:
            # 给一个占位符避免 SDK 报 missing key；调用时若服务真的要 key，会返回 401
            api_key = "EMPTY"
        self.model = model
        self.base_url = base_url
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def stream_chat(self, messages: list[dict]) -> Iterator[str]:
        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    yield content
        except AuthenticationError as e:
            raise ProviderError(f"鉴权失败（API Key 错误）：{e}")
        except RateLimitError as e:
            raise ProviderError(f"达到调用频率/配额上限：{e}", retryable=True)
        except APIConnectionError as e:
            raise ProviderError(f"无法连接 API（检查 base_url 与网络）：{e}", retryable=True)
        except APIError as e:
            raise ProviderError(f"API 返回错误：{e}")
        except Exception as e:
            raise ProviderError(f"调用失败：{e}")
