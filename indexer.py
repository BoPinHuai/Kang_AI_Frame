import re
import json
import chromadb
from chromadb.config import Settings
from pathlib import Path
from embedder import get_model
from loader import load_document

SUPPORTED = {".pdf", ".docx", ".md"}
_CHROMA_SETTINGS = Settings(anonymized_telemetry=False)

# 切分/索引策略版本号——影响 chunk 内容或 embedding 输入时递增
CHUNK_VERSION = "semantic_v3"
MAX_CHUNK = 300
MIN_CHUNK = 30


# ── 语义切分 ────────────────────────────────────────────
def chunk_text(text: str, ext: str = "", max_chunk: int = MAX_CHUNK) -> list[str]:
    parts = _split_markdown(text) if ext == ".md" else _split_paragraphs(text)
    return _merge_and_limit(parts, max_chunk)


def _split_markdown(text: str) -> list[str]:
    parts = re.split(r'(?m)(?=^#{1,3} )', text)
    return [p.strip() for p in parts if p.strip()]


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r'\n{2,}', text)
    return [p.strip() for p in parts if p.strip()]


def _split_by_sentence(text: str, max_size: int) -> list[str]:
    sentences = re.split(r'(?<=[。！？\.\!\?])\s*', text)
    return _merge_and_limit(sentences, max_size)


def _fallback_window(text: str, size: int = MAX_CHUNK, overlap: int = 80) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return [c for c in chunks if c.strip()]


def _merge_and_limit(parts: list[str], max_size: int) -> list[str]:
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) > max_size:
            if buf:
                chunks.append('\n\n'.join(buf))
                buf, buf_len = [], 0
            sub = _split_by_sentence(part, max_size)
            for s in sub:
                if len(s) > max_size:
                    chunks.extend(_fallback_window(s, max_size))
                else:
                    chunks.append(s)
        elif buf_len + len(part) > max_size:
            if buf:
                chunks.append('\n\n'.join(buf))
            buf = [part]
            buf_len = len(part)
        else:
            buf.append(part)
            buf_len += len(part)

    if buf:
        chunks.append('\n\n'.join(buf))

    return [c for c in chunks if len(c.strip()) >= MIN_CHUNK]


# ── Meta 帮助函数 ────────────────────────────────────────
def _load_meta(meta_path: str) -> dict:
    p = Path(meta_path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_meta(meta_path: str, meta: dict):
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ── 主索引函数 ────────────────────────────────────────────
def build_index(library_folder: str, db_path: str, meta_path: str,
                force: bool = False, progress_cb=None):
    """增量建索引：只处理新增或改动的文件。force=True 时强制全量重建。

    Returns: (indexed_count, skipped_count)
    """
    def _p(cur, tot, msg):
        if progress_cb:
            try:
                progress_cb(cur, tot, msg)
            except Exception:
                pass

    from settings import load_settings
    _max_chunk = load_settings().get("max_chunk", MAX_CHUNK)

    _p(0, 0, "扫描文件中…")
    library = Path(library_folder)
    client = chromadb.PersistentClient(path=db_path, settings=_CHROMA_SETTINGS)

    meta = {} if force else _load_meta(meta_path)

    # 切分策略升级 → 自动全量重建
    if not force and meta.get("__chunk_version__") != CHUNK_VERSION:
        print(f"  切分策略升级（→ {CHUNK_VERSION}），自动全量重建")
        force = True
        meta = {}

    if force:
        try:
            client.delete_collection("knowledge_base")
        except Exception:
            pass
        collection = client.create_collection("knowledge_base")
        print("  强制重建：已清空旧索引")
    else:
        try:
            collection = client.get_collection("knowledge_base")
        except Exception:
            collection = client.create_collection("knowledge_base")

    # 扫描 library/ 下全部文件
    current_files: dict[str, float] = {}
    for f in library.rglob("*"):
        if not f.is_file():
            continue
        if f.name.startswith("~$") or f.name.startswith("."):
            continue
        if f.suffix.lower() in SUPPORTED:
            rel = str(f.relative_to(library)).replace("\\", "/")
            current_files[rel] = f.stat().st_mtime

    if not current_files:
        print("  library/ 为空，无文件可索引")
        _save_meta(meta_path, {"__chunk_version__": CHUNK_VERSION})
        return 0, 0

    # 删除已不存在文件的旧索引块
    to_remove = [p for p in meta if not p.startswith("__") and p not in current_files]
    for rel_path in to_remove:
        old_ids = set(meta[rel_path].get("chunk_ids", []))
        if old_ids:
            try:
                collection.delete(ids=list(old_ids))
            except Exception:
                pass
        del meta[rel_path]
        print(f"  移除旧索引: {rel_path}")

    # 找出需要（重新）索引的文件
    to_index = [
        rel for rel, mtime in current_files.items()
        if abs(meta.get(rel, {}).get("mtime", 0) - mtime) > 0.01
    ]
    skipped = len(current_files) - len(to_index)

    if not to_index:
        print(f"  全部 {len(current_files)} 个文件均已是最新，无需重新索引")
        _p(0, 0, f"全部 {len(current_files)} 个文件已最新")
        return 0, skipped

    total = len(to_index)
    print(f"  需索引 {total} 个文件（{skipped} 个已是最新，跳过）")
    _p(0, total, f"需索引 {total} 个文件")

    embed_model = get_model()
    indexed = 0
    for fi, rel_path in enumerate(to_index):
        _p(fi, total, f"正在索引（{fi+1}/{total}）：{rel_path}")
        abs_path = library / rel_path
        ext = Path(rel_path).suffix.lower()
        try:
            text = load_document(str(abs_path))
            if not text.strip():
                print(f"  跳过（内容为空）: {rel_path}")
                continue

            old_ids = set(meta.get(rel_path, {}).get("chunk_ids", []))
            if old_ids:
                try:
                    collection.delete(ids=list(old_ids))
                except Exception:
                    pass

            chunks = chunk_text(text, ext, _max_chunk)
            safe_id = (rel_path.replace("/", "__")
                                .replace("\\", "__")
                                .replace(" ", "_")
                                .replace(".", "_"))
            chunk_ids = [f"{safe_id}_{ci}" for ci in range(len(chunks))]

            embeddings = embed_model.encode(chunks, show_progress_bar=False).tolist()
            metadatas = [{"source": rel_path}] * len(chunks)

            batch_size = 100
            for bi in range(0, len(chunks), batch_size):
                collection.add(
                    documents=chunks[bi:bi + batch_size],
                    embeddings=embeddings[bi:bi + batch_size],
                    ids=chunk_ids[bi:bi + batch_size],
                    metadatas=metadatas[bi:bi + batch_size],
                )

            meta[rel_path] = {"mtime": current_files[rel_path], "chunk_ids": chunk_ids}
            indexed += 1
            print(f"  已索引: {rel_path}（{len(chunks)} 块）")

        except Exception as e:
            print(f"  错误 [{rel_path}]: {e}")

    meta["__chunk_version__"] = CHUNK_VERSION
    _save_meta(meta_path, meta)
    print(f"  索引更新完成：新建/更新 {indexed}，跳过 {skipped}")
    _p(total, total, f"完成：新增/更新 {indexed}，跳过 {skipped}")
    return indexed, skipped
