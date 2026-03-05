import os
from filelock import FileLock
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

def warm():
    cache = os.getenv("FASTEMBED_CACHE_PATH", "/mnt/models/fastembed_cache")
    os.makedirs(cache, exist_ok=True)

    lock_path = os.path.join(cache, ".warmup.lock")
    with FileLock(lock_path, timeout=600):
        # Import inside lock so multiple tasks don't race downloads

        # Use the exact models your pipeline uses
        emb = TextEmbedding("nomic-ai/nomic-embed-text-v1.5")
        # Trigger download
        list(emb.embed(["warmup"]))

        reranker = TextCrossEncoder("BAAI/bge-reranker-base")
        # Trigger download
        reranker.rerank("warmup", ["warmup doc"])

if __name__ == "__main__":
    warm()