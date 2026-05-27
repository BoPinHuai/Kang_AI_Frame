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
        raise ValueError(f"不支持的格式: {suffix}（支持 .pdf .docx .md）")


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
    """递归读取文件夹内所有支持的文档，filename 为相对路径，返回 [{filename, text}, ...]"""
    supported = {".pdf", ".docx", ".md"}
    docs = []
    root = Path(folder)
    for file in root.rglob("*"):
        if not file.is_file():
            continue
        if file.name.startswith("~$") or file.name.startswith("."):
            continue
        if file.suffix.lower() in supported:
            rel = str(file.relative_to(root)).replace("\\", "/")
            print(f"  正在读取: {rel}")
            text = load_document(str(file))
            if text.strip():
                docs.append({"filename": rel, "text": text})
            else:
                print(f"  警告: {rel} 内容为空，已跳过")
    return docs
