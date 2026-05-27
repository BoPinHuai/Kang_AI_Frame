from abc import ABC, abstractmethod
from typing import Iterator


class ProviderError(Exception):
    """Provider 层统一异常（网络错误、鉴权失败、模型未就绪等）"""
    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class ChatProvider(ABC):
    """所有聊天模型 Provider 的统一接口。"""

    name: str = "base"

    @abstractmethod
    def stream_chat(self, messages: list[dict]) -> Iterator[str]:
        """流式对话。

        Args:
            messages: [{"role": "system"/"user"/"assistant", "content": "..."}]

        Yields:
            str: 模型每次产出的文本片段（chunk）

        Raises:
            ProviderError: 调用失败时抛出。retryable=True 表示调用方可重试。
        """
        ...

    def test(self) -> tuple[bool, str]:
        """测试连接是否可用。返回 (ok, message)。"""
        try:
            chunks = []
            for c in self.stream_chat([
                {"role": "user", "content": "hi"}
            ]):
                chunks.append(c)
                if sum(len(x) for x in chunks) > 10:
                    break
            return True, "连接正常"
        except ProviderError as e:
            return False, str(e)
        except Exception as e:
            return False, f"未知错误：{e}"
