# Kang AI Frame — 本地知识库问答系统

> 上传文档，用 AI 回答问题。数据留在本地，模型自由选择。

---

## 功能一览

| 功能 | 说明 |
|------|------|
| 📄 文档支持 | PDF、Word (.docx)、Markdown、Excel (.xlsx)、CSV |
| 🔍 语义检索 | ChromaDB 向量检索，支持 `@文件夹` 限定范围 |
| 🤖 模型自由 | 本地 Ollama 或任意 OpenAI 兼容 API（DeepSeek / Kimi / OpenAI 等） |
| ⚡ 智能路由 | 问候 / 闲聊自动跳过检索，直接回答 |
| 💬 多轮对话 | 携带历史上下文，流式输出 |
| 🗂️ 知识库管理 | 文件夹浏览、文件预览、来源定位 |
| 🖥️ 桌面客户端 | pywebview 封装，双击运行，无需开浏览器 |
| 🔒 访问密码 | 可选开启，防止他人访问 |

---

## 快速开始

### 方式一：网页版（推荐）

**环境要求**：Python 3.10+

```bash
# 1. 克隆项目
git clone https://github.com/BoPinHuai/Kang_AI_Frame.git
cd Kang_AI_Frame

# 2. 安装依赖
pip install -r requirements.txt

# 3. 复制配置文件
cp config.example.json config.json

# 4. 启动
python api.py
```

浏览器打开 [http://127.0.0.1:8000](http://127.0.0.1:8000)

> 首次启动会自动下载 Embedding 模型（约 100MB）。  
> 国内网络较慢时，打开 `embedder.py` 取消镜像注释那行。

---

### 方式二：桌面客户端

```bash
pip install -r requirements.txt
python app_launcher.py
```

自动弹出独立窗口，无需打开浏览器。  
发布版本的 `.exe` 文件见 [Releases](https://github.com/BoPinHuai/Kang_AI_Frame/releases)。

---

## 配置模型

首次启动后编辑 `config.json`（由 `config.example.json` 自动生成）：

**本地 Ollama（默认）**
```json
{
  "active": "rag",
  "providers": {
    "rag": { "type": "ollama", "model": "qwen2.5:7b", "base_url": "http://localhost:11434" }
  }
}
```

**DeepSeek API**
```json
{
  "active": "deepseek",
  "providers": {
    "deepseek": { "type": "openai", "model": "deepseek-chat", "base_url": "https://api.deepseek.com/v1", "api_key": "你的Key" }
  }
}
```

也可以直接在前端「模型设置」面板里切换，无需手动编辑文件。

---

## 使用方式

1. **上传文档**：点击输入框左侧 `+` 按钮上传，或直接拖拽
2. **提问**：直接在输入框输入问题，系统自动检索相关内容
3. **限定范围**：输入 `@文件夹名` 只在指定目录内检索，或在知识库面板点击「聚焦」
4. **查看来源**：回答下方的来源卡片显示引用片段和位置，点击可定位到知识库

---

## 项目结构

```
api.py              FastAPI 服务入口，所有 REST 路由
chat.py             流式问答 + 智能路由（rag_mode）
retriever.py        向量检索，cosine 相似度
indexer.py          增量索引（仅更新变动文件）
loader.py           文档解析（PDF/DOCX/MD/XLSX/CSV → 分段列表）
embedder.py         Embedding 模型单例（BAAI/bge-small-zh-v1.5）
settings.py         全局参数中心（top_k、num_ctx 等）
config.py           Provider 管理
providers/          Ollama / OpenAI 适配层
app_launcher.py     桌面客户端启动器（pywebview）
app_launcher.spec   PyInstaller 打包配置
static/             前端（单文件 HTML + JS）
library/            放你的文档（gitignored）
db/                 向量数据库（gitignored）
hf-cache/           模型缓存（gitignored）
```

---

## 常见问题

**Q：不想用 Ollama，可以只用 API 吗？**  
可以。在 `config.json` 里把 `active` 改为 openai 类型的 provider，填入 API Key 即可，无需安装 Ollama。

**Q：首次运行下载模型太慢？**  
打开 `embedder.py`，找到被注释掉的 `HF_ENDPOINT` 那行，取消注释即可走国内镜像。

**Q：如何重建索引？**  
前端「知识库」面板 → 右上角「更新索引」按钮，或删除 `db/` 和 `db_meta.json` 后重启。

**Q：多人可以同时使用吗？**  
可以。启动时将 host 改为 `0.0.0.0`，同局域网内的人访问你的 IP:8000 即可。  
建议同时在「个人资料」里开启访问密码。

---

## 依赖说明

| 包 | 用途 |
|----|------|
| fastapi + uvicorn | Web 服务框架 |
| chromadb | 向量数据库 |
| sentence-transformers | 文本 Embedding |
| pypdf / python-docx / openpyxl | 文档解析 |
| ollama / openai | 模型调用 |
| pywebview | 桌面客户端窗口 |

---

## License

MIT
