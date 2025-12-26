from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

# =========================
# RETRIEVAL PROMPTS
# =========================

QUERY_EXPANSION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You generate search queries for a retrieval system."),
        (
            "user",
            "Rewrite the user question into {n} diverse search queries.\n"
            "Rules:\n"
            "- Keep each query short (<= 15 words)\n"
            "- Use different wording or synonyms\n"
            "- Expand acronyms if helpful\n"
            "- Output ONLY a JSON array of strings\n\n"
            "Question: {question}",
        ),
    ]
)



RERANK_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a strict relevance ranker for retrieval."),
        (
            "user",
            "Given a question and passages, score each passage for relevance.\n"
            "Return ONLY a JSON array of objects: "
            '[{"index": <int>, "score": <float 0-10>}] in the SAME order.\n\n'
            "Question: {question}\n\n"
            "Passages:\n{passages}",
        ),
    ]
)

# =========================
# GENERATION PROMPTS
# =========================

RAG_TEMPLATE = """Answer the question based ONLY on the following context:
{context}

Question: {question}
"""

@traceable
def build_rag_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(RAG_TEMPLATE)