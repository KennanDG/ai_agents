import sys

from ai_agents.rag.query import answer, answer_langgraph
from ai_agents.rag.settings import RagSettings


def main():
    if len(sys.argv) < 2:
        print('Usage: rag_query "your question"')
        raise SystemExit(2)

    settings = RagSettings()

    print("qdrant_url:", settings.qdrant_url)
    print("collection:", f"{settings.collection_name}-{settings.namespace}")
    print("embedding_model:", settings.embedding_model)
    print("chat_model:", settings.chat_model)
    print("query_model:", settings.query_model)
    print("rerank_model:", settings.rerank_model)
    print("k:", settings.k)
    print("candidate_k:", settings.candidate_k)
    print("k_per_query:", settings.k_per_query)
    print("rrf_k:", settings.rrf_k)
    print("n_query_expansions:", settings.n_query_expansions)

    q = sys.argv[1]
    print(answer_langgraph(q, settings))

if __name__ == "__main__":
    main()