import os
import json
import time

import faiss
import numpy as np
from dotenv import load_dotenv

# Import GenAI SDK
from google import genai

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment")

# Initialize GenAI client for Gemini Developer API
client = genai.Client(api_key=API_KEY)

# File paths (use raw strings on Windows)
CHUNKS_FILE = r"C:\Users\Zoro\Desktop\McD_RAG\data\chunks.jsonl"
INDEX_FILE  = r"C:\Users\Zoro\Desktop\McD_RAG\data\menu_index.faiss"
META_FILE   = r"C:\Users\Zoro\Desktop\McD_RAG\data\chunks_meta.json"

def load_chunks(path):
    """Load serialized chunks from JSONL."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def embed_texts(texts):
    """Batch-embed text chunks using Gemini embedding model."""
    embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.models.embed_content(
            model="gemini-embedding-001",
            contents=batch
        )
        embeddings.extend([e.values for e in resp.embeddings])
        time.sleep(1)  # throttle to respect rate limits
    return embeddings

def main():
    print(f"Loading chunks from {CHUNKS_FILE} …")
    chunks = load_chunks(CHUNKS_FILE)
    texts = [chunk["text"] for chunk in chunks]
    print(f"Embedding {len(texts)} chunks…")

    embeddings = embed_texts(texts)
    if not embeddings:
        raise RuntimeError("No embeddings were generated. Check your API key and chunk data.")

    dim = len(embeddings[0])
    xb  = np.array(embeddings, dtype="float32")

    print(f"Building FAISS index ({dim} dims) with {len(xb)} vectors…")
    index = faiss.IndexFlatL2(dim)
    index.add(xb)

    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
    faiss.write_index(index, INDEX_FILE)
    print(f"Saved FAISS index to {INDEX_FILE}")

    meta = [chunk.get("metadata", {}) for chunk in chunks]
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved chunk metadata to {META_FILE}")

if __name__ == "__main__":
    main()
