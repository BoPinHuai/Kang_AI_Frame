import re
import time
import chromadb
from chromadb.config import Settings
from embedder import get_model
from settings import load_settings

_CHROMA_SETTINGS = Settings(anonymized_telemetry=False)
_collection = None


def reset_collection():
    global _collection
    _collection = None


def _get_collection(db_path: str):
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=db_path, settings=_CHROMA_SETTINGS)
        _collection = client.get_collection("knowledge_base")
    return _collection


def _contains_chinese(text: str) -> bool:
    return bool(re.search(r'[一-鿿]', text))


def _translate_to_english(question: str) -> str:
    """用当前激活的 LLM 将中文问题翻译成英文，失败时返回原文。"""
    try:
        from config import get_active_provider
        provider = get_active_provider()
        prompt = (
            "Translate the following question to English. "
            "Output only the translated text, nothing else.\n\n"
            + question
        )
        result = ""
        for chunk in provider.stream_chat([{"role": "user", "content": prompt}]):
            result += chunk
        result = result.strip()
        print(f"[跨语言] 原文：{question!r} → 译文：{result!r}")
        return result if result else question
    except Exception as e:
        print(f"[跨语言] 翻译失败，使用原文检索：{e}")
        return question


def retrieve(question: str, db_path: str = "./db",
             scope: str = None) -> list[dict]:
    s = load_settings()
    top_k = s["top_k_scoped"] if scope else s["top_k"]

    # 中文问题 → 翻译成英文再检索
    query = _translate_to_english(question) if _contains_chinese(question) else question

    t0 = time.perf_counter()
    try:
        col = _get_collection(db_path)
    except Exception:
        return []

    if col.count() == 0:
        return []

    t1 = time.perf_counter()
    embedding = get_model().encode([query], normalize_embeddings=True).tolist()
    t2 = time.perf_counter()

    results = col.query(
        query_embeddings=embedding,
        n_results=min(top_k, col.count()),
        include=["documents", "metadatas", "distances"],
    )
    t3 = time.perf_counter()

    min_score = s.get("min_score", 50)
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        relevance = round((1 - dist) * 100)
        if relevance < min_score:
            continue   # 相关度不足，丢弃
        source = meta["source"]
        if scope and not source.startswith(scope.rstrip("/") + "/") and source != scope.rstrip("/"):
            continue
        chunks.append({"text": doc, "source": source,
                       "loc": meta.get("loc", ""), "distance": dist})

    print(f"[耗时] embed={t2-t1:.3f}s  chroma={t3-t2:.3f}s  检索总计={t3-t0:.3f}s")
    return chunks
