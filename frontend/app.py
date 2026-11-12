import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Assistant", page_icon="📚")
st.title("📚 RAG Document Assistant")
st.caption("Ask questions about your documents. Answers come only from the indexed files.")



def call_api(method: str, path: str, timeout: int = 120, **kwargs):
    """Call the backend. Returns (data, error_message), one of them is None."""
    try:
        response = requests.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
    except requests.exceptions.ConnectionError:
        return None, f"Cannot reach the backend at {API_URL}. Is it running?"
    except requests.exceptions.Timeout:
        return None, "The backend took too long to respond."
    
    if response.status_code == 200:
        return response.json(), None
    
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    if isinstance(detail, list):
        detail = "; ".join(item.get("msg", "invalid input") for item in detail)
    return None, f"Error {response.status_code}: {detail}"


def render_extras(msg: dict):
    """Show confidence and source citations"""
    confidence = msg.get("confidence")
    if confidence:
        st.caption(f"Confidence: **{confidence}**")
    sources = msg.get("sources") or []
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for s in sources:
                st.markdown(f"**{s['source']}** (distance {s['distance']})")
                st.write(s["text"])
                st.divider()




# -- Sidebar--
with st.sidebar:
    st.header("Controls")
    if st.button("Re-index Documents", use_container_width=True):
        with st.spinner("Indexing documents (the first run downloads an embedding model)..."):
            data, error = call_api("POST", "/ingest", timeout=600)
        if error:
            st.error(error)
        else:
            st.success(data["message"])

    st.divider()   
    st.subheader("System status")
    health, error = call_api("GET", "/health", timeout=5)
    if error:
        st.error(error)
    else:
        st.write(f"ChromaDB: **{health['chromadb']}**")
        st.write(f"Ollama: **{health['ollama']}**")
        st.write(f"Chunks indexed: **{health['documents']}")

    stats, _ = call_api("GET", "/stats", timeout=5)
    if stats:
        st.write(f"Model: **{stats['ollama_model']}")



#  -- chat --
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_extras(msg)


question = st.chat_input("Ask a question about your documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            data, error = call_api("POST", "/ask", json={"question": question})
            if error:
                st.error(error)
                st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {error}"})
            else:
                msg = {
                    "role": "assistant",
                    "content": data["answer"],
                    "sources": data["sources"],
                    "confidence": data["confidence"],
                }
                st.markdown(msg["content"])
                render_extras(msg)
                st.session_state.messages.append(msg)

