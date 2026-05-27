import re
import time
from retriever import retrieve
from config import get_active_provider
from providers import ProviderError
from settings import load_settings


# ── 闲聊/元问题检测（rag_mode="auto" 时使用）──────────────────────────────
# 匹配到这些模式 → 跳过 RAG 检索，直接让模型回答
_CONV_PATTERNS = re.compile(
    r'(^(你好|嗨|哈喽|hello|hi|hey|早上好|下午好|晚上好|早安|晚安)\b'  # 问候
    r'|谢谢|感谢|非常感谢|太感谢|好的(，|。|！|$)|明白了|懂了|知道了|没问题|好棒|厉害'  # 回应
    r'|再见|拜拜|bye|结束对话|退出'                                       # 告别
    r'|你是谁|你叫什么|你的名字|你是什么模型|你用的什么模型|你是哪个模型|你是哪款'  # AI 身份
    r'|你能做什么|你有什么功能|你会什么|你能帮(我|我们)做什么|你的功能'     # AI 能力
    r'|我刚才(说|问|讲)|刚才我(说|问|讲)|我之前(说|问|讲)|之前我(说|问|讲)'  # 回溯对话
    r'|上(一|个|面|条).*(问题|问的)|之前(问|说|讲)|我们(刚才|之前|聊)'
    r'|总结.*(对话|聊天)|对话.*总结|聊了什么|问了什么|我问过什么)',
    re.IGNORECASE,
)


def _is_conversational(question: str) -> bool:
    """启发式判断：该问题是否不需要知识库检索。"""
    q = question.strip()
    # 极短且不含信息检索关键词 → 视为闲聊
    if len(q) <= 8 and not re.search(r'(什么是|如何|怎么|为什么|哪个|哪些|介绍|说明)', q):
        return True
    return bool(_CONV_PATTERNS.search(q))


# ── 核心流式问答 ──────────────────────────────────────────────────────────
def ask_stream(question: str, db_path: str = "./db",
               history: list = None, scope: str = None):
    s = load_settings()
    t_start = time.perf_counter()

    rag_mode = s.get("rag_mode", "auto")

    # 决定是否执行检索
    do_retrieve = True
    if rag_mode == "never":
        do_retrieve = False
    elif rag_mode == "auto" and not scope:
        # 有 scope（@文件）时强制检索；否则智能判断
        if _is_conversational(question):
            do_retrieve = False

    if do_retrieve:
        chunks = retrieve(question, db_path, scope=scope)
    else:
        chunks = []
        print(f"[路由] rag_mode={rag_mode} → 跳过检索，直接对话")

    yield "sources", chunks

    t_retrieved = time.perf_counter()

    if chunks:
        context = "\n\n".join([f"【片段{i+1}】\n{c['text']}" for i, c in enumerate(chunks)])
        if scope:
            user_message = f"以下是文件「{scope}」中检索到的内容片段：\n\n{context}\n\n请根据以上内容回答问题：{question}"
        else:
            user_message = f"参考资料：\n{context}\n\n问题：{question}"
    else:
        user_message = question

    input_chars = len(s["system_prompt"]) + len(user_message)
    print(f"[耗时] 检索={t_retrieved-t_start:.3f}s  prompt总字数={input_chars}")

    messages = [{"role": "system", "content": s["system_prompt"]}]
    if s["use_history"] and history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    try:
        provider = get_active_provider()
    except ProviderError as e:
        yield "error", f"模型未配置：{e}"
        return

    try:
        first_token = True
        token_count = 0
        t_llm_start = time.perf_counter()
        for content in provider.stream_chat(messages):
            if first_token:
                print(f"[耗时] 首token={time.perf_counter()-t_llm_start:.3f}s")
                first_token = False
            token_count += 1
            yield "chunk", content
        print(f"[耗时] 生成完毕={time.perf_counter()-t_llm_start:.3f}s  tokens≈{token_count}  全程={time.perf_counter()-t_start:.3f}s")
        yield "done", ""
    except ProviderError as e:
        yield "error", f"模型调用失败：{e}"
    except Exception as e:
        yield "error", f"未知错误：{e}"
