import os
from pathlib import Path

# 国内访问 HuggingFace 较慢时，取消下一行注释使用镜像加速
# os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

CACHE_DIR = Path(__file__).parent / "hf-cache"
CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print("正在加载向量模型...")
        try:
            _model = SentenceTransformer(
                _MODEL_NAME, cache_folder=str(CACHE_DIR), local_files_only=True,
            )
            print("向量模型加载完成（离线模式）")
        except Exception as e:
            print(f"本地缓存不可用，尝试联网下载…（{e}）")
            _model = SentenceTransformer(_MODEL_NAME, cache_folder=str(CACHE_DIR))
            print("向量模型加载完成（已下载到本地缓存）")
    return _model
