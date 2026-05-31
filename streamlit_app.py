# streamlit_app.py
# Streamlit frontend for the Viennabase RAG chatbot.
#
# This app acts as a thin HTTP client — it sends questions to the FastAPI backend
# and displays the responses. No direct access to the RAG pipeline or documents.
#
# Default backend: http://127.0.0.1:8000/ask (local FastAPI instance)
# Authentication: optional x-api-key header, configurable in the sidebar
#
# Prerequisites:
#   - FastAPI server running: uvicorn api:app --reload
#   - .env with APP_API_URL and APP_API_KEY (optional)
#
# Start with:
#   streamlit run streamlit_app.py

from __future__ import annotations
import os
import requests
from typing import Dict, Any, List

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DEFAULT_API_URL = os.getenv("APP_API_URL", "http://127.0.0.1:8000/ask")
DEFAULT_API_KEY = os.getenv("APP_API_KEY", "")


def call_api(
    question: str,
    chat_history: List[Dict[str, str]],
    api_url: str,
    api_key: str,
    debug: bool = False,
    timeout_s: int = 30,
) -> Dict[str, Any]:
    """
    Send a question and chat history to the /ask endpoint and return a normalized response.

    Return schema:
      {
        "ok": bool,
        "answer": str,
        "sources": List[{"source": str, "label": str}],
        "error": str | None
      }
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key

    payload = {
        "question": question,
        "chat_history": chat_history,
        "debug": debug,
    }

    try:
        resp = requests.post(api_url, json=payload, headers=headers, timeout=timeout_s)
    except requests.exceptions.RequestException as e:
        return {"ok": False, "answer": "", "sources": [], "error": f"Connection error: {e}"}

    if resp.status_code != 200:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        return {
            "ok": False,
            "answer": "",
            "sources": [],
            "error": f"HTTP {resp.status_code}: {detail}",
        }

    try:
        data = resp.json()
    except ValueError:
        return {"ok": False, "answer": "", "sources": [], "error": "Invalid JSON in response."}

    answer = data.get("answer", "")
    sources = data.get("sources", []) or []

    norm_sources: List[Dict[str, str]] = []
    for s in sources:
        if isinstance(s, dict):
            norm_sources.append(
                {"source": s.get("source", ""), "label": s.get("label", "")}
            )

    return {
        "ok": True,
        "answer": answer,
        "sources": norm_sources,
        "error": None,
        "retrieval_query": data.get("retrieval_query", ""),
        "debug_history_text": data.get("debug_history_text", ""),
        "debug_context_preview": data.get("debug_context_preview", ""),
        "debug_no_context": data.get("debug_no_context", None),
    }


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Viennabase RAG Chatbot", page_icon="💬", layout="centered")

st.title("💬 Viennabase RAG Chatbot")
st.caption(
    "Ask questions about the AGBs, Aufnahmerichtlinien, Heimstatut, or FAQ. "
    "Answers are based exclusively on the local knowledge base."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Sidebar: settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")
    api_url = st.text_input(
        "API URL", value=DEFAULT_API_URL, help="FastAPI endpoint for /ask"
    )
    api_key = st.text_input(
        "API Key (x-api-key header)",
        value=DEFAULT_API_KEY,
        type="password",
        help="Leave empty if API key authentication is disabled.",
    )
    st.markdown("---")
    st.markdown(
        "💡 **Note:** Start the API locally with:\n\n"
        "`uvicorn api:app --reload`"
    )
    debug_mode = st.checkbox(
        "Show debug output",
        value=False,
        help="Displays retrieval query and context preview for troubleshooting.",
    )

    if st.button("Reset chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Chat history display
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            sources = msg.get("sources", [])
            if sources:
                st.markdown("**Sources**")
                for s in sources:
                    label = f" [{s['label']}]" if s.get("label") else ""
                    st.markdown(f"- {s.get('source', '')}{label}")

# ---------------------------------------------------------------------------
# New message input
# ---------------------------------------------------------------------------

question = st.chat_input("Ask a question about Viennabase...")

if question:
    user_message = {"role": "user", "content": question.strip()}
    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(user_message["content"])

    chat_history_payload = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            result = call_api(
                question=question.strip(),
                chat_history=chat_history_payload,
                api_url=api_url.strip(),
                api_key=api_key.strip(),
                debug=debug_mode,
            )

        if not result["ok"]:
            st.error(result["error"] or "Unknown error.")
        else:
            answer_text = result["answer"] or "The API returned no answer."
            st.markdown(answer_text)

            if result["sources"]:
                st.markdown("**Sources**")
                for s in result["sources"]:
                    label = f" [{s['label']}]" if s.get("label") else ""
                    st.markdown(f"- {s.get('source', '')}{label}")

            if debug_mode:
                with st.expander("Debug"):
                    st.markdown("**Retrieval Query**")
                    st.code(result.get("retrieval_query", ""))

                    st.markdown("**No Context**")
                    st.write(result.get("debug_no_context"))

                    st.markdown("**Chat History (formatted)**")
                    st.text(result.get("debug_history_text", ""))

                    st.markdown("**Context Preview**")
                    st.text(result.get("debug_context_preview", ""))

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_text,
                "sources": result["sources"],
            })
