import sys

from ai_agents.rag.query import answer
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
    print("k:", settings.k)

    q = sys.argv[1]
    print(answer(q, settings))

if __name__ == "__main__":
    main()