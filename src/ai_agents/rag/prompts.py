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



COLLECTION_ROUTER_TEMPLATE = """You are a router for a RAG system. Choose the best Qdrant collections to search.

Return JSON with:
- preferred_collections: an ordered list of up to {max_fallbacks} collection base names
- reason: short explanation

Rules:
- Only choose from the allowed list.
- Always include "{default_collection}" as the last element in the list as a fallback.
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



DOCUMENTS_GRADER_TEMPLATE = """You are grading whether retrieved documents are good context to answer the question.

Return ONLY valid JSON:
{{
  "score": 0 or 1,
  "reason": "short explanation"
}}

Rules:
- score=1 if the documents are clearly relevant and contain enough information to answer.
- score=0 if documents are irrelevant, too thin, or totally mismatched to the question.

Question:
{question}

Documents (snippets):
{context}
"""

@traceable
def build_docs_grader_prompt() -> ChatPromptTemplate:
    """Build the prompt used for collection routing."""
    return ChatPromptTemplate.from_template(DOCUMENTS_GRADER_TEMPLATE)




# =========================
# VERIFICATION PROMPTS
# =========================

GRADER_TEMPLATE = """You are a strict verifier for a Retrieval-Augmented Generation (RAG) system.

Your job: decide if the ANSWER is acceptable given the CONTEXT.

Key idea:
- FAIL only for MATERIAL problems (material hallucinations or not answering the question).
- Do NOT fail for minor paraphrasing, reasonable high-level synthesis, or generic glue text.

Definitions:
- "Supported" means the claim is explicitly stated OR is a direct, minimal paraphrase of the context.
- "Material hallucination" means the answer introduces specific facts, entities, steps, numbers, or mechanisms
  that are not in the context AND they matter to the correctness of the answer.

Scoring:
PASS (score=1) if ALL are true:
1) The answer addresses the user's question (even at a high level if the question is broad).
2) The answer contains NO material hallucinations.
3) Most of the substantive content is supported by the context.

FAIL (score=0) if ANY are true:
A) The answer contains a material hallucination.
B) The answer is mostly irrelevant or does not answer the question.
C) The answer is primarily "cannot be determined / not provided" with little to no grounded help.

Notes:
- If the context is somewhat thin but the answer is still grounded and helpful, you may PASS.
  In that case, mention in the reason that retrieval could be improved, but do not fail.
- Do not penalize the answer for minor framing language (e.g., "one approach is...") as long as it
  does not introduce material new facts.

On FAIL (score=0), also produce a rewritten version of the ORIGINAL_QUESTION to help retrieval.


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

RAG_TEMPLATE = """You are an expert assistant capable of extracting precise information from documents.

Your goal is to answer the user's question using ONLY the provided context snippets below.

Guidelines:
1. Grounding: Answer strictly based on the provided context. Do not use outside knowledge.
2. Uncertainty: If the context does not contain enough information to answer the question fully, state what is missing or say "The provided context does not contain information about..."
3. Tone: Be professional, direct, and concise. Avoid conversational filler; but maintain warm, friendly tone.
4. Completeness: Address all parts of the user's question if supported by the context.

Context:
{context}

Question:
{question}
"""

@traceable
def build_rag_prompt() -> ChatPromptTemplate:
    """Build the prompt used for answer generation."""
    return ChatPromptTemplate.from_template(RAG_TEMPLATE)