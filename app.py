import os
import streamlit as st
import requests
import faiss
import numpy as np
import pickle
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "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")

headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}

# Load sentence transformer for embedding
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS Vector DB Initialization
dim = 384  # Dimension of embeddings
index_file = "faiss_index.bin"
history_file = "chat_history.pkl"

# Initialize FAISS and Chat History
if os.path.exists(index_file) and os.path.exists(history_file):
    try:
        index = faiss.read_index(index_file)
        with open(history_file, "rb") as f:
            chat_history = pickle.load(f)
    except Exception as e:
        st.sidebar.warning(f"⚠️ Error loading history: {e}")
        index = faiss.IndexFlatL2(dim)
        chat_history = []
else:
    index = faiss.IndexFlatL2(dim)
    chat_history = []


def generate_response(prompt):
    """Fetch response from Hugging Face API."""
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})

    if response.status_code != 200:
        return f"Error: {response.status_code} - {response.text}"

    try:
        output = response.json()
        return output[0].get("generated_text", "No response generated. Try again.")
    except (KeyError, IndexError, TypeError):
        return "Error: Invalid response format from API."


def save_to_faiss(user_input, bot_response):
    """Save chat to FAISS index with duplicate check."""
    global index, chat_history
    chat_text = f"User: {user_input} | AI: {bot_response}"
    embedding = embed_model.encode([chat_text])

    # Check for duplicates
    if chat_text in chat_history:
        return  

    index.add(np.array(embedding, dtype=np.float32))
    chat_history.append(chat_text)

    # Save FAISS index and history
    faiss.write_index(index, index_file)
    with open(history_file, "wb") as f:
        pickle.dump(chat_history, f)


def retrieve_similar_queries(user_input, top_k=3):
    """Retrieve past queries similar to the user input."""
    if len(chat_history) == 0:
        return ["No previous chat history."]

    query_vector = embed_model.encode([user_input])
    D, I = index.search(np.array(query_vector, dtype=np.float32), top_k)
    similar_chats = [chat_history[i] for i in I[0] if i < len(chat_history)]
    
    return similar_chats


def main():
    st.set_page_config(page_title="AI ChatBot", page_icon="🤖")
    st.header("🤖 AI Knowledge ChatBot ")

    # Sidebar - Chat History
    st.sidebar.title("📜 Chat History")
    if chat_history:
        for chat in reversed(chat_history[-5:]):  # Show last 5 messages
            user_msg, ai_msg = chat.split("| AI: ")
            st.sidebar.markdown(f"🧑‍💻 **{user_msg}**<br>🤖 {ai_msg}", unsafe_allow_html=True)
    else:
        st.sidebar.text("No history yet.")

    messages = st.container()

    if prompt := st.chat_input("Say something"):
        messages.chat_message("user").write(prompt)

        # Generate AI response
        response = generate_response(prompt)
        messages.chat_message("assistant").write(f"AiChat: {response}")

        # Save to FAISS database
        save_to_faiss(prompt, response)

        # Retrieve similar past queries
        similar_chats = retrieve_similar_queries(prompt)
        st.subheader("🔍 Related Past Conversations:")
        for chat in similar_chats:
            st.text(chat)


if __name__ == "__main__":
    main()
