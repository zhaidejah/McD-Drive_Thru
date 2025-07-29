import streamlit as st
import os
import json
import faiss
import numpy as np
from google import genai
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

INDEX_FILE = "data/menu_index.faiss"
META_FILE = "data/chunks_meta.json"
CHUNKS_FILE = "data/chunks.jsonl"

@st.cache_resource
def load_index():
    return faiss.read_index(INDEX_FILE)

@st.cache_data
def load_chunks():
    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def embed_query(query):
    resp = client.models.embed_content(model="gemini-embedding-001", contents=[query])
    return np.array(resp.embeddings[0].values, dtype="float32")

def chat_response(context, query):
    prompt = f"""You are a helpful assistant answering specifically about McDonald's Menu items using this context:

{context}

User: {query}
Answer:"""
    response = client.models.chat(model="gemini-2.0-flash", prompt=prompt, temperature=0.3)
    return response.text

def main():
    st.title("McDonald's Menu RAG System with Gemini & FAISS")

    index = load_index()
    chunks = load_chunks()

    query = st.text_input("Ask a question about McDonald's menu:")
    if query:
        q_emb = embed_query(query)
        D, I = index.search(np.array([q_emb]), 5)
        retrieved = [chunks[i]["text"] for i in I[0]]
        context = "\n\n".join(retrieved)
        answer = chat_response(context, query)
        st.subheader("Answer")
        st.write(answer)

if __name__ == "__main__":
    main()
