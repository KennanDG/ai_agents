import sys
from ai_agents.rag.settings import RagSettings
from ai_agents.rag.query import answer

def main():
    if len(sys.argv) < 2:
        print('Usage: rag_query "your question"')
        raise SystemExit(2)

    settings = RagSettings()
    q = sys.argv[1]
    print(answer(q, settings))

if __name__ == "__main__":
    main()