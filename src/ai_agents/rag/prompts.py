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
            "- Keep each query concise (<= 1-2 sentences)\n"
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



COLLECTION_ROUTER_TEMPLATE = """
You are a router for a RAG system. Choose the best Qdrant collections to search.

Return JSON with:
- preferred_collections: an ordered list of up to {max_fallbacks} collection base names
- reason: short explanation

Rules:
- Only choose from the allowed list.
- Always include "{default_collection}" somewhere in the list as a fallback.
- Order from most likely to least likely.

Allowed collections:
{available}

Question:
{question}
"""

@traceable
def build_collection_router_prompt() -> ChatPromptTemplate:
    """Build the prompt used for collection routing."""
    return ChatPromptTemplate.from_template(COLLECTION_ROUTER_TEMPLATE)




# =========================
# VERIFICATION PROMPTS
# =========================

GRADER_TEMPLATE = """You are a strict verifier for a Retrieval-Augmented Generation (RAG) system.

Your job: decide if the ANSWER is acceptable given the CONTEXT.

You must grade using these rules:

PASS (score=1) only if ALL are true:
1) The answer is directly supported by the context (no hallucinations).
2) The answer addresses the user's question.
3) The context contains enough information to answer the question with real substance.
   - The answer should include specific facts/details grounded in the context.
   - If the context does not mention the key entities/requirements needed to answer, it is NOT sufficient.

FAIL (score=0) if ANY are true:
A) The answer includes claims not found in the context.
B) The answer is irrelevant or does not actually answer the question.
C) The answer indicates missing context or uncertainty due to missing context, e.g.:
   - "There is no mention of ..."
   - "Not provided in the context"
   - "I don't have enough information"
   - "Cannot be determined from the provided documents"
   In these cases: the retrieval did not supply enough context, so FAIL.

On FAIL (score=0), you must also produce a rewritten version of the ORIGINAL QUESTION
to help the retrieval system fetch different documents on the next attempt.

Rewriting rules:
- Keep the user's intent the same.
- Add 1-3 clarifying details that improve retrievability (synonyms, likely related terms, roles, context hints).
- If the question is ambiguous, add disambiguation angles
- Do NOT invent facts. You may only add generic disambiguation prompts.

Given:
CONTEXT:
{context}

ORIGINAL_QUESTION:
{original_question}

QUESTION_USED_THIS_ATTEMPT:
{question}

ANSWER:
{answer}

Return ONLY valid JSON with exactly these keys:
{{
  "score": 1 or 0,
  "reason": "<short reason>",
  "rewritten_question": "<string; required. If score=1, return the ORIGINAL_QUESTION unchanged.>"
}}
"""

@traceable
def build_grader_prompt() -> ChatPromptTemplate:
    """Build the prompt used for answer verification (groundedness + relevance).

    Notes:
        - The downstream verifier chain expects JSON output.
        - Keep the prompt deterministic: temperature=0 and explicit JSON schema.
    """
    return ChatPromptTemplate.from_template(GRADER_TEMPLATE)



# =========================
# GENERATION PROMPTS
# =========================

RAG_TEMPLATE = """Answer the question based ONLY on the following context:
{context}

Question: {question}
"""

@traceable
def build_rag_prompt() -> ChatPromptTemplate:
    """Build the prompt used for answer generation."""
    return ChatPromptTemplate.from_template(RAG_TEMPLATE)