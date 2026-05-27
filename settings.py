"""
全局可调参数中心。

  通过前端「参数设置」面板修改，自动保存到 config.json["settings"]。
  也可直接修改下方 DEFAULTS 字典（重启后以 DEFAULTS 为基准补全缺失项）。

参数使用位置速查：
  system_prompt  → chat.py            AI 风格/角色定义
  use_history    → chat.py            是否携带多轮对话上下文
  rag_mode       → chat.py            检索模式："auto"智能判断 / "always"始终检索 / "never"直接对话
  top_k          → retriever.py       无 scope 时检索片段数
  top_k_scoped   → retriever.py       有 scope（@文件）时检索片段数
  num_ctx        → providers/ollama_provider.py   LLM 上下文窗口（token）
  num_predict    → providers/ollama_provider.py   LLM 最大输出 token 数
  max_chunk      → indexer.py         文档分块最大字符数（改后需重建索引）
"""
import json
from pathlib import Path

BASE_DIR    = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.json"

DEFAULTS: dict = {
    # ── chat.py ──────────────────────────────────────────────────────────────
    "system_prompt": (
        "你是一个智能助手。根据参考资料回答用户问题，"
        "资料不相关时用自己的知识回答。回答详细完整，充分展开说明。"
    ),
    "use_history": True,        # True = 携带最近几轮历史；False = 每次独立问答
    # "auto"   → 智能判断：问候/闲聊/元问题自动跳过检索，直接对话
    # "always" → 始终检索知识库，无论提问内容
    # "never"  → 始终直接对话，完全跳过 RAG 检索
    "rag_mode": "auto",

    # ── retriever.py ─────────────────────────────────────────────────────────
    "top_k": 1,                 # 无 scope 时从向量库取几个片段喂给 LLM（1-5）
    "top_k_scoped": 6,          # 有 scope 时取几个片段（建议 4-10）

    # ── providers/ollama_provider.py ─────────────────────────────────────────
    "num_ctx": 1024,            # LLM 上下文窗口大小，越小 prefill 越快（512-8192）
    "num_predict": 2048,        # LLM 单次最大输出 token 数（128-4096）

    # ── indexer.py（修改后须在前端点「更新索引」才能生效）────────────────────
    "max_chunk": 300,           # 文档分块最大字符数，越小检索越精准（100-800）
}


def load_settings() -> dict:
    """读取当前参数，缺失字段自动补默认值。"""
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
        return {**DEFAULTS, **cfg.get("settings", {})}
    except Exception:
        return dict(DEFAULTS)


def save_settings(incoming: dict) -> dict:
    """保存参数到 config.json['settings']，只保留已知字段，返回完整参数。"""
    merged = {k: incoming.get(k, v) for k, v in DEFAULTS.items()}
    # bool 字段防止前端传字符串 "true"
    merged["use_history"] = bool(merged["use_history"])
    # rag_mode 只允许三个合法值
    if merged["rag_mode"] not in ("auto", "always", "never"):
        merged["rag_mode"] = "auto"
    # 数值字段防止越界
    merged["top_k"]       = max(1, int(merged["top_k"]))
    merged["top_k_scoped"]= max(1, int(merged["top_k_scoped"]))
    merged["num_ctx"]     = max(512, int(merged["num_ctx"]))
    merged["num_predict"] = max(128, int(merged["num_predict"]))
    merged["max_chunk"]   = max(100, int(merged["max_chunk"]))
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}
    cfg["settings"] = merged
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return merged
