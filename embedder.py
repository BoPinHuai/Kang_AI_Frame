import os
import sys
import threading
from pathlib import Path

# 国内访问 HuggingFace 较慢时，取消下一行注释使用镜像加速
# os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 打包后模型缓存放在 exe 同级目录，避免写入只读的 _internal/
_BASE = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).parent
CACHE_DIR = _BASE / "hf-cache"
CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

_MODEL_NAME = "BAAI/bge-m3"
_model = None
_model_lock = threading.Lock()   # 防止多线程重复加载/下载


def get_model():
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:   # 拿到锁后再次确认
            from sentence_transformers import SentenceTransformer
            print("正在加载向量模型...")
            _model = SentenceTransformer(_MODEL_NAME, cache_folder=str(CACHE_DIR))
            print("向量模型加载完成（已缓存到本地）")
    return _model
