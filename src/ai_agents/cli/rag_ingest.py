import sys

from ai_agents.rag.ingest import ingest_files
from ai_agents.rag.settings import RagSettings
from ai_agents.rag.embeddings import build_fastembed_embeddings


def main():
    if len(sys.argv) < 2:
        print("Usage: rag_ingest <file1> [file2 ...]")
        raise SystemExit(2)

    settings = RagSettings()
    embeddings = build_fastembed_embeddings(settings.embedding_model, settings.chunk_size)
    test_embedding = embeddings.embed_query("test")
    model_dimension = len(test_embedding)

    print("qdrant_url:", settings.qdrant_url)
    print("collection:", f"{settings.collection_name}-{settings.namespace}")
    print("embedding_model:", settings.embedding_model)
    print("chat_model:", settings.chat_model)
    print("k:", settings.k)
    print("embedding_model dimension", model_dimension)
    print()

    n = ingest_files(sys.argv[1:], settings)
    
    print(f"Ingested {n} chunks into vectorstore")
    print()
    print()

if __name__ == "__main__":
    main()