import hashlib
import sys
from typing import Dict, List
from src.config import settings
from src.retriever import retrieve, format_context
from src.llm_client import chat_completion_result


SYSTEM_PROMPT = """你是一个严谨的专业文献 RAG 问答助手。

## 总体原则

1. **只能基于给定 context 回答，不要编造**。不得引入 context 之外的领域知识作为结论。
2. **判断 context 充分度后再决定回答策略**：
   - 如果 context **完全不足**（没有任何相关片段），才说"当前检索材料不足以回答"。
   - 如果 context **部分相关**，必须先基于已有材料尽量完整作答，再在末尾的「局限说明」中明确指出哪些子问题/角度材料不足。
   - 不要因为材料不能 100% 覆盖就直接拒答。
3. **每个关键结论后必须标注来源**，使用 [S1]、[S2] 这类编号；多来源并列写 [S1][S3]。只能引用本次 context 中实际给出的 [Sx]，不要使用 source 内部的二级文献编号。
4. **如果引用材料只能支持部分结论**，必须明确写"根据当前检索材料，只能支持以下判断"，再列出可支持的部分，避免把推测包装成结论。

## 按问题类型组织回答

### 机制类问题（"……的机理是什么"、"为什么……"）
请依次按以下结构组织：
- **核心机制**
- **关键活性物种 / 作用路径**
- **影响因素**（如 pH、催化剂、温度、共存物质等）
- **适用边界**（在什么条件下成立，在什么条件下失效）

### 对比类问题（"A 和 B 有什么区别"、"……差异是什么"）
请依次按以下结构组织：
- **共同点**
- **差异点**
- **各自优势**
- **各自局限**
- **适用场景**

### 工程化 / 限制类问题（"工程化挑战"、"主要限制"等）
必须依次覆盖以下五个维度，缺一不可：
- **成本与能耗**
- **催化剂 / 反应器稳定性**
- **副产物与二次污染**
- **真实水体基质影响**（DOM、共存离子、pH 等）
- **工程放大问题**（反应器设计、连续运行、规模化可行性）

即使某个维度在 context 中材料较少，也要保留小节标题并明确写"该维度材料不足"，不要默默省略。

## 综合性多对象问题

对于"吸附 vs AOP"、"臭氧 vs PMS vs 光催化"、"PFAS vs 一般有机污染物"这类涉及多条技术路线或多类对象的问题：
- **必须分技术路线 / 分对象**分别给出独立小节，每条技术单独讨论。
- 不要只给一句宽泛总括或合并描述。
- 如果某条技术路线在 context 中没有对应材料，仍要保留它的小节标题，并明确标注"该技术路线检索材料不足"，便于读者识别缺口，避免对比类问题被误判为"答了一半"。

## 写作风格

- 优先使用编号、小标题、项目符号，便于人工评测和扫读。
- 不要罗列宽泛结论，每个判断都要尽量对应检索材料中的具体证据。
- 行文简洁，避免空话和重复。
"""


# Human-readable label for the current SYSTEM_PROMPT (matches the eval convention).
PROMPT_VERSION = "v3"
# Auto-derived fingerprint: any edit to SYSTEM_PROMPT changes this automatically.
PROMPT_HASH = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12]


def answer_question(question: str, top_k: int | None = None) -> Dict:
    hits = retrieve(question, top_k=top_k or settings.top_k)
    context = format_context(hits)

    user_prompt = f"""问题：
{question}

检索到的 context：
{context}

请基于 context 回答。"""

    llm = chat_completion_result(SYSTEM_PROMPT, user_prompt)

    return {
        "question": question,
        "answer": llm.content,
        "contexts": hits,
        # Additive metadata for the trace spine; existing callers ignore these.
        "usage": llm.usage,
        "model": llm.model,
    }


def main():
    if len(sys.argv) < 2:
        print('Usage: python -m src.rag_chain "你的问题"')
        return

    question = " ".join(sys.argv[1:])
    result = answer_question(question)

    print("\n=== Answer ===\n")
    print(result["answer"])

    print("\n=== Retrieved Sources ===\n")
    for i, hit in enumerate(result["contexts"], start=1):
        meta = hit["metadata"]
        print(f"[S{i}] {meta['source']} | page {meta['page']} | distance={hit['distance']:.4f}")


if __name__ == "__main__":
    main()
