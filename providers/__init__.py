from .base import ChatProvider, ProviderError
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider

__all__ = ["ChatProvider", "ProviderError", "OllamaProvider", "OpenAIProvider", "build_provider"]


def build_provider(profile: dict) -> ChatProvider:
    """根据 config 中的单个 profile 字典构造对应 Provider 实例。

    profile 示例：
        {"type": "ollama", "model": "qwen2.5:7b", "base_url": "http://localhost:11434"}
        {"type": "openai", "model": "deepseek-chat", "base_url": "https://api.deepseek.com/v1", "api_key": "sk-..."}
    """
    ptype = (profile.get("type") or "").lower()
    if ptype == "ollama":
        return OllamaProvider(
            model=profile.get("model", "qwen2.5:7b"),
            base_url=profile.get("base_url") or None,
        )
    if ptype in ("openai", "openai-compatible", "openai_compatible"):
        return OpenAIProvider(
            model=profile.get("model", "gpt-4o-mini"),
            base_url=profile.get("base_url") or "https://api.openai.com/v1",
            api_key=profile.get("api_key") or "",
        )
    raise ProviderError(f"未知的 provider 类型: {ptype}")
