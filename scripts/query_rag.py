import os
import json
import faiss
import numpy as np
from google import genai
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment")

client = genai.Client(api_key=API_KEY)

INDEX_FILE = r"C:\Users\Zoro\Desktop\McD_RAG\data\menu_index.faiss"
META_FILE = r"C:\Users\Zoro\Desktop\McD_RAG\data\chunks_meta.json"
CHUNKS_FILE = r"C:\Users\Zoro\Desktop\McD_RAG\data\chunks.jsonl"

def load_faiss_index(path):
    return faiss.read_index(path)

def load_metadata(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_chunks(path):
    return [json.loads(line) for line in open(path, "r", encoding="utf-8")]

def embed_query(query):
    resp = client.models.embed_content(
        model="gemini-embedding-001",
        contents=[query]
    )
    return resp.embeddings[0].values

def chat_with_context(context: str, user_query: str) -> str:
    # Create a chat session
    chat = client.chats.create(model="gemini-2.0-flash")
    # Send the combined context + user query as a single message
    prompt = (
        "You are a helpful assistant answering questions about McDonald's menu using this context:\n\n"
        f"{context}\n\nUser: {user_query}"
    )
    response = chat.send_message(prompt)
    return response.text

def main():
    print("Loading FAISS index and metadata...")
    index = load_faiss_index(INDEX_FILE)
    metadata = load_metadata(META_FILE)
    chunks = load_chunks(CHUNKS_FILE)

    while True:
        user_query = input("\nEnter your McDonald's menu question (or 'exit' to quit): ")
        if user_query.lower() in ("exit", "quit"):
            break

        q_emb = np.array([embed_query(user_query)], dtype="float32")
        D, I = index.search(q_emb, 5)
        retrieved = [chunks[i]["text"] for i in I[0]]
        combined_context = "\n\n".join(retrieved)

        answer = chat_with_context(combined_context, user_query)
        print(f"\nAnswer:\n{answer}")

if __name__ == "__main__":
    main()
