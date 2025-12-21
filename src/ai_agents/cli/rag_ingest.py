import sys
from ai_agents.rag.settings import RagSettings
from ai_agents.rag.ingest import ingest_files

def main():
    if len(sys.argv) < 2:
        print("Usage: rag_ingest <file1> [file2 ...]")
        raise SystemExit(2)

    settings = RagSettings()
    n = ingest_files(sys.argv[1:], settings)
    print(f"Ingested {n} chunks into {settings.persist_dir}")

if __name__ == "__main__":
    main()