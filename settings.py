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
  min_score      → retriever.py       相关度最低门槛（0-100），低于此分的片段丢弃
  num_ctx        → providers/ollama_provider.py   LLM 上下文窗口（token）
  num_predict    → providers/ollama_provider.py   LLM 最大输出 token 数
  max_chunk      → indexer.py         文档分块最大字符数（改后需重建索引）
"""
import sys
import json
from pathlib import Path

# PyInstaller 打包后 __file__ 指向 _internal/ 临时目录，用户数据要放在 exe 同级目录
BASE_DIR    = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.json"

DEFAULTS: dict = {
    # ── chat.py ──────────────────────────────────────────────────────────────
    "system_prompt": (
        "你是一个智能助手。根据参考资料回答用户问题，"
        "资料不相关时用自己的知识回答。回答详细完整，充分展开说明。"
        "所有数学表达式（包括括号内、句子中间的简短公式）必须使用 LaTeX 格式："
        "行内公式用 $...$，独立公式用 $$...$$。"
        "严禁用 √、²、σ 等 Unicode 符号或纯文本写法（如 E/(3(1-2v))）替代 LaTeX。"
    ),
    "use_history": True,        # True = 携带最近几轮历史；False = 每次独立问答
    # "auto"   → 智能判断：问候/闲聊/元问题自动跳过检索，直接对话
    # "always" → 始终检索知识库，无论提问内容
    # "never"  → 始终直接对话，完全跳过 RAG 检索
    "rag_mode": "auto",

    # ── 用户档案（前端「个人资料」编辑）──────────────────────────────────────
    "user_name": "朋友",         # 首屏问候 / 侧栏显示的名字，可随意修改
    "user_avatar": "",          # 头像 data URL（base64），空则显示首字母
    "app_name": "知识库",        # 侧栏顶部显示的应用名称，可自定义
    "lock_enabled": False,      # True = 启用访问密码，False = 直接进入
    "lock_password": "",        # 访问密码（明文，仅限本地部署使用）

    # ── retriever.py ─────────────────────────────────────────────────────────
    "top_k": 3,                 # 无 scope 时从向量库取几个片段喂给 LLM（1-5）
    "top_k_scoped": 6,          # 有 scope 时取几个片段（建议 4-10）
    "min_score": 50,            # 相关度最低门槛（0-100），低于此分的片段丢弃

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
    """保存参数到 config.json['settings']。
    在「当前已保存值」之上做局部覆盖——只更新 incoming 里出现的已知字段，
    这样参数面板与个人资料面板分别提交时不会互相清空对方的字段。
    """
    merged = load_settings()  # 当前值（含默认补全）
    for k in DEFAULTS:
        if k in incoming:
            merged[k] = incoming[k]
    # bool 字段防止前端传字符串 "true"
    merged["use_history"] = bool(merged["use_history"])
    # rag_mode 只允许三个合法值
    if merged["rag_mode"] not in ("auto", "always", "never"):
        merged["rag_mode"] = "auto"
    # 数值字段防止越界
    merged["top_k"]       = max(1, int(merged["top_k"]))
    merged["top_k_scoped"]= max(1, int(merged["top_k_scoped"]))
    merged["min_score"]   = max(0, min(100, int(merged.get("min_score", 50))))
    merged["num_ctx"]     = max(512, int(merged["num_ctx"]))
    merged["num_predict"] = max(128, int(merged["num_predict"]))
    merged["max_chunk"]   = max(100, int(merged["max_chunk"]))
    # 用户档案
    name = str(merged.get("user_name", "")).strip()[:24]
    merged["user_name"] = name or "朋友"
    merged["user_avatar"] = str(merged.get("user_avatar", "") or "")
    merged["lock_enabled"] = bool(merged.get("lock_enabled", False))
    # 只在 incoming 里有 lock_password 时才更新（避免覆盖为空）
    if "lock_password" in incoming:
        merged["lock_password"] = str(incoming["lock_password"] or "")
    else:
        merged["lock_password"] = str(merged.get("lock_password", "") or "")
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}
    cfg["settings"] = merged
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return merged
