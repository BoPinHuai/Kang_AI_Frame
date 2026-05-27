# 本地 RAG 知识库问答系统 — 完整指导

## 项目目标

上传 PDF / Word / Markdown 文档，用本地 AI 对话问答，数据不出电脑。

---

## 技术栈

| 组件 | 工具 | 作用 |
|------|------|------|
| 本地 LLM | Ollama + Qwen2.5:7b | 生成回答 |
| 向量 Embedding | sentence-transformers | 文字转向量 |
| 向量数据库 | ChromaDB | 存储和检索 |
| 文档解析 | pypdf / python-docx | 读取文件 |
| 界面（可选） | gradio | 网页对话界面 |

---

## 环境准备

### 1. 安装 Ollama

去 https://ollama.com 下载安装包，安装后终端运行：

```bash
# 下载中文模型（约 4.5GB，放外置硬盘见下方配置）
ollama pull qwen2.5:7b

# 验证是否成功
ollama run qwen2.5:7b
# 能对话就成功，Ctrl+D 退出
```

### 2. 外置硬盘存放模型（可选）

```bash
# Windows
setx OLLAMA_MODELS "D:\ollama-models"

# Mac/Linux，加入 ~/.zshrc 或 ~/.bashrc
export OLLAMA_MODELS="/Volumes/MyDisk/ollama-models"

# 设置完重启 Ollama 再 pull 模型
```

### 3. 安装 Python 依赖

```bash
pip install ollama chromadb sentence-transformers pypdf python-docx gradio
```

---

## 项目结构

```
rag-project/
├── main.py          # 入口，命令行问答
├── loader.py        # 文档读取（PDF/Word/MD）
├── indexer.py       # 切块 + 向量化 + 存入 ChromaDB
├── retriever.py     # 检索相关片段
├── chat.py          # 调用 Ollama 生成回答
├── app.py           # Gradio 网页界面（可选）
├── docs/            # 放你的文档
│   ├── 合同.pdf
│   ├── 说明.docx
│   └── 笔记.md
└── db/              # ChromaDB 自动生成，可指向外置盘
```

---

## 代码实现

### loader.py — 统一读取三种格式

```python
import os
from pathlib import Path
import pypdf
from docx import Document

def load_document(file_path: str) -> str:
    """读取 PDF / Word / Markdown，统一返回纯文本"""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(file_path)
    elif suffix == ".docx":
        return _load_docx(file_path)
    elif suffix == ".md":
        return _load_md(file_path)
    else:
        raise ValueError(f"不支持的格式: {suffix}")

def _load_pdf(file_path: str) -> str:
    reader = pypdf.PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def _load_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def _load_md(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def load_all_docs(folder: str) -> list[dict]:
    """读取文件夹内所有支持的文档"""
    supported = {".pdf", ".docx", ".md"}
    docs = []
    for file in Path(folder).rglob("*"):
        if file.suffix.lower() in supported:
            print(f"正在读取: {file.name}")
            text = load_document(str(file))
            docs.append({"filename": file.name, "text": text})
    return docs
```

---

### indexer.py — 切块 + 向量化 + 存库

```python
import chromadb
from sentence_transformers import SentenceTransformer

# 向量模型，首次运行自动下载（约 500MB）
EMBED_MODEL = SentenceTransformer("BAAI/bge-small-zh-v1.5")  # 中文优化

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """把长文本切成小块，相邻块有重叠避免信息断裂"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def build_index(docs: list[dict], db_path: str = "./db"):
    """把所有文档向量化存入 ChromaDB"""
    client = chromadb.PersistentClient(path=db_path)
    
    # 如果集合已存在则清空重建
    try:
        client.delete_collection("knowledge_base")
    except:
        pass
    collection = client.create_collection("knowledge_base")

    all_chunks = []
    all_ids = []
    all_metadata = []

    for doc in docs:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{doc['filename']}_{i}")
            all_metadata.append({"source": doc["filename"]})

    print(f"共 {len(all_chunks)} 个文本块，正在向量化...")
    embeddings = EMBED_MODEL.encode(all_chunks).tolist()

    # 分批写入避免内存问题
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        collection.add(
            documents=all_chunks[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            ids=all_ids[i:i+batch_size],
            metadatas=all_metadata[i:i+batch_size],
        )
    print("✅ 索引构建完成")
```

---

### retriever.py — 检索最相关片段

```python
import chromadb
from sentence_transformers import SentenceTransformer

EMBED_MODEL = SentenceTransformer("BAAI/bge-small-zh-v1.5")

def retrieve(question: str, db_path: str = "./db", top_k: int = 5) -> list[dict]:
    """根据问题检索最相关的文本片段"""
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection("knowledge_base")

    question_embedding = EMBED_MODEL.encode([question]).tolist()
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k
    )

    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"text": doc, "source": meta["source"]})
    return chunks
```

---

### chat.py — 调用 Ollama 回答

```python
import ollama
from retriever import retrieve

SYSTEM_PROMPT = """你是一个知识库问答助手。
请严格根据提供的参考资料回答用户问题。
如果资料中没有相关信息，请直接说"文档中没有找到相关内容"，不要编造答案。
回答要简洁、准确，并注明信息来自哪个文档。"""

def ask(question: str, db_path: str = "./db", model: str = "qwen2.5:7b") -> str:
    # 1. 检索相关片段
    chunks = retrieve(question, db_path)
    
    # 2. 拼接上下文
    context = "\n\n".join([
        f"【来源: {c['source']}】\n{c['text']}" 
        for c in chunks
    ])
    
    # 3. 构建 prompt
    user_message = f"""参考资料：
{context}

用户问题：{question}"""

    # 4. 调用本地模型
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
    )
    return response["message"]["content"]
```

---

### main.py — 命令行入口

```python
from loader import load_all_docs
from indexer import build_index
from chat import ask
import os

DB_PATH = "./db"          # 改成外置盘路径如 "D:/my-knowledge-base"
DOCS_FOLDER = "./docs"    # 放你的文档

def main():
    # 第一次运行：建索引
    if not os.path.exists(DB_PATH):
        print("🔨 首次运行，正在建立知识库索引...")
        docs = load_all_docs(DOCS_FOLDER)
        build_index(docs, DB_PATH)
    else:
        print("✅ 知识库已存在，直接加载")

    print("\n💬 知识库已就绪，开始问答（输入 quit 退出）\n")

    while True:
        question = input("你：").strip()
        if question.lower() in ("quit", "exit", "退出"):
            break
        if not question:
            continue
        
        print("AI：", end="", flush=True)
        answer = ask(question, DB_PATH)
        print(answer)
        print()

if __name__ == "__main__":
    main()
```

---

### app.py — Gradio 网页界面（可选）

```python
import gradio as gr
from chat import ask

def respond(question, history):
    answer = ask(question)
    return answer

demo = gr.ChatInterface(
    fn=respond,
    title="📚 本地知识库问答",
    description="基于你的私有文档，完全本地运行",
    examples=["这份合同的违约金是多少？", "请总结文档的主要内容"],
)

if __name__ == "__main__":
    demo.launch()
```

运行后浏览器打开 http://localhost:7860 即可。

---

## 运行步骤

```bash
# 1. 把文档放入 docs/ 文件夹

# 2. 首次建立索引（之后不用重复）
python main.py

# 3. 或者启动网页界面
python app.py
```

---

## 外置硬盘配置

把 `main.py` 里的路径改掉：

```python
# Windows
DB_PATH = "D:/my-knowledge-base/db"
DOCS_FOLDER = "D:/my-knowledge-base/docs"

# Mac
DB_PATH = "/Volumes/MyDisk/knowledge-base/db"
DOCS_FOLDER = "/Volumes/MyDisk/knowledge-base/docs"
```

---

## 常见问题

**Q: PDF 乱码或提取不到文字？**
> 说明是扫描版 PDF，需要 OCR。安装 `pip install pytesseract`，另外需要装 Tesseract 软件本体。

**Q: 模型回答很慢？**
> 集显正常，7b 模型大约 5-15 秒/次。可换更小的 `qwen2.5:1.5b` 提速。

**Q: 想重新建索引（新增了文档）？**
> 删除 `db/` 文件夹，重新运行 `main.py` 即可。

**Q: 回答不准确？**
> 调整 `chunk_size`（试试 300 或 800），或增大 `top_k`（检索更多片段）。

---

## 升级方向（之后可以做）

- [ ] 支持增量添加文档（不用重建全部索引）
- [ ] 显示回答来源的具体页码
- [ ] 支持多轮对话记忆
- [ ] 接入 Claude API 替换本地模型（回答更准）
- [ ] 文档管理界面（上传、删除、预览）
