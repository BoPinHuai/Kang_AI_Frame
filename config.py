"""配置管理：读写 config.json，构造当前 active provider。

config.json 不进 git；首次启动自动从 config.example.json 复制。
"""
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Optional

from providers import ChatProvider, ProviderError, build_provider

BASE_DIR    = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.json"
EXAMPLE_PATH = BASE_DIR / "config.example.json"

# 默认配置：本地 Ollama，不需要 key 就能直接跑
DEFAULT_CONFIG = {
    "active": "local",
    "providers": {
        "local": {
            "type": "ollama",
            "model": "qwen2.5:7b",
            "base_url": "http://localhost:11434",
        }
    },
}


def _ensure_config() -> None:
    """确保 config.json 存在。优先从 example 复制，否则用 DEFAULT_CONFIG。"""
    if CONFIG_PATH.exists():
        return
    if EXAMPLE_PATH.exists():
        shutil.copy(EXAMPLE_PATH, CONFIG_PATH)
    else:
        save_config(DEFAULT_CONFIG)


def load_config() -> dict:
    """读取当前 config，缺失字段自动补齐。"""
    _ensure_config()
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}
    # 合并默认值，避免缺字段
    if "providers" not in cfg or not isinstance(cfg["providers"], dict):
        cfg["providers"] = {}
    if not cfg["providers"]:
        cfg["providers"] = deepcopy(DEFAULT_CONFIG["providers"])
    if "active" not in cfg or cfg["active"] not in cfg["providers"]:
        cfg["active"] = next(iter(cfg["providers"]))
    return cfg


def save_config(cfg: dict) -> None:
    """保存 config（保留 api_key 等敏感字段，原样写盘）。"""
    global _provider_cache
    _provider_cache = None
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def public_view(cfg: dict) -> dict:
    """对外暴露的视图：遮蔽 api_key，只显示是否已设置。"""
    view = {"active": cfg["active"], "providers": {}}
    for name, p in cfg["providers"].items():
        item = {k: v for k, v in p.items() if k != "api_key"}
        item["has_key"] = bool(p.get("api_key"))
        view["providers"][name] = item
    return view


def merge_update(cfg: dict, incoming: dict) -> dict:
    """把前端传上来的新 config 合并到现有 config。

    规则：
      - active 若提供则覆盖
      - providers 整体替换为前端版本
      - 若某个 provider 的 api_key 为空字符串，保留原有值（避免前端遮蔽后误清空）
    """
    new_cfg = deepcopy(cfg)
    if "active" in incoming:
        new_cfg["active"] = incoming["active"]
    if "providers" in incoming and isinstance(incoming["providers"], dict):
        merged_providers = {}
        for name, p in incoming["providers"].items():
            old = cfg["providers"].get(name, {})
            item = dict(p)
            # 若前端没传 api_key 或传了空串，保留旧值
            if not item.get("api_key") and old.get("api_key"):
                item["api_key"] = old["api_key"]
            # 去掉前端可能带回来的 has_key 字段
            item.pop("has_key", None)
            merged_providers[name] = item
        new_cfg["providers"] = merged_providers
    # 保证 active 合法
    if new_cfg["active"] not in new_cfg["providers"]:
        new_cfg["active"] = next(iter(new_cfg["providers"]))
    return new_cfg


_provider_cache: Optional["ChatProvider"] = None


def get_active_provider(cfg: Optional[dict] = None) -> ChatProvider:
    """根据 config 构造当前 active 的 Provider 实例（结果缓存，config 变更时自动失效）。"""
    global _provider_cache
    if cfg is None:
        if _provider_cache is not None:
            return _provider_cache
        cfg = load_config()
        name = cfg["active"]
        profile = cfg["providers"].get(name)
        if not profile:
            raise ProviderError(f"找不到 active provider：{name}")
        _provider_cache = build_provider(profile)
        return _provider_cache
    name = cfg["active"]
    profile = cfg["providers"].get(name)
    if not profile:
        raise ProviderError(f"找不到 active provider：{name}")
    return build_provider(profile)


def get_provider_by_name(name: str, cfg: Optional[dict] = None) -> ChatProvider:
    """按名字取 provider，主要给测试连接用。"""
    cfg = cfg or load_config()
    profile = cfg["providers"].get(name)
    if not profile:
        raise ProviderError(f"找不到 provider：{name}")
    return build_provider(profile)
