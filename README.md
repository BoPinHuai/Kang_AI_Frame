# 隔离知识库

支持上传 PDF / Word / Markdown 文档，通过向量检索 + 大模型回答问题。

## 功能

- 文档上传与管理（PDF、DOCX、Markdown）
- 向量语义检索，支持 `@文件名` 精准定位
- 多模型支持：本地 Ollama 或任意 OpenAI 兼容 API（DeepSeek、Kimi 等）
- 流式回答，多轮对话
- 智能路由：问候/闲聊自动跳过检索，直接回答
- 前端可调所有参数（检索数量、回复长度、AI 风格等）

## 快速开始

### 1. 环境要求

- Python 3.10+ （之后会更新简单的启动）
- 选择一种模型方式就行：
  - **本地**：安装 [Ollama](https://ollama.com) 并拉取模型
  - **外部 API**：准备 DeepSeek / OpenAI / Kimi 等 API Key

### 2. 安装依赖

- 已经为大家准备好版本清单啦
- 执行下面就好了
```
pip install -r requirements.txt
```

### 3. 配置

```bash
cp config.example.json config.json
```

编辑 `config.json`，将 `active` 设为你要用的 provider 名称，填入对应的模型和 Key。

**本地 Ollama 示例：**
```json
{
  "active": "rag",
  "providers": {
    "rag": {
      "type": "ollama",
      "model": "qwen2.5:7b",
      "base_url": "http://localhost:11434"
    }
  }
}
```

**外部 API 示例（DeepSeek）：**
```json
{
  "active": "deepseek",
  "providers": {
    "deepseek": {
      "type": "openai",
      "model": "deepseek-chat",
      "base_url": "https://api.deepseek.com/v1",
      "api_key": "你的Key"
    }
  }
}
```

### 4. 启动

```bash
python api.py
```

浏览器打开 [http://127.0.0.1:8000](http://127.0.0.1:8000)

> **首次启动**会自动下载 embedding 模型（约 100MB）。大家如果在国内网络较慢时，打开 `embedder.py` 取消镜像那行的注释。

## 使用说明

- 前端页面大家自行探索

## 项目结构

```
api.py          # FastAPI 服务入口
chat.py         # 流式问答 + 智能路由
retriever.py    # 向量检索
indexer.py      # 文档索引（增量）
embedder.py     # Embedding 模型加载
settings.py     # 全局参数中心
config.py       # Provider 管理
providers/      # Ollama / OpenAI 适配层
static/         # 前端页面
library/        # 放你的文档（gitignored）
```
