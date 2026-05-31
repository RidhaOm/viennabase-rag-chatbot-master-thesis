# chatbot.py
# Core RAG pipeline for the Viennabase chatbot.
# Loads FAQ content and PDF documents, builds a Chroma vector store,
# and answers questions using a retriever + LLM setup.
# Guardrails are handled in separate modules (input_guardrails.py, output_guardrails.py).

from __future__ import annotations
import os
import re
import shutil
import unicodedata
import requests
from typing import List, Tuple, Dict, Any

from dotenv import load_dotenv
from bs4 import BeautifulSoup

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser

from prompts import VIENNABASE_RAG_PROMPT, HISTORY_AWARE_RETRIEVAL_PROMPT
from input_guardrails import validate_query
from output_guardrails import is_safe_text


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FAQ_URL = "https://viennabase.at/faq/"
PDFS = [
    ("data/AGBs.pdf", "AGBs"),
    ("data/Aufnahmerichtlinien.pdf", "Aufnahmerichtlinien"),
    ("data/Heimstatut.pdf", "Heimstatut"),
]
PERSIST_DIR = "chroma_db"
EMBED_MODEL = "text-embedding-3-small"
RESET_VECTORSTORE_ON_START = True

# Set by build_rag_chain(); defined here so api.py can import without errors.
RETRIEVER = None  # type: ignore

# Distance threshold for evidence filtering (lower = more similar).
# Chunks with distance > threshold are discarded.
EVIDENCE_DISTANCE_THRESHOLD = 1.10

# Set to False to disable the threshold (ablation study: no_threshold config).
USE_EVIDENCE_THRESHOLD = True


def load_env() -> None:
    """Load .env and check that OPENAI_API_KEY is present."""
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY is not set.")


# ---------------------------------------------------------------------------
# Data loading: FAQ and PDF documents
# ---------------------------------------------------------------------------

def scrape_faq(url: str = FAQ_URL) -> List[Document]:
    """
    Fetch the FAQ page and extract question/answer pairs as LangChain documents.
    Returns an empty list if the page is unavailable or has an unexpected structure.
    """
    faqs: List[Document] = []
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except Exception as e:
        print(f"Warning: Could not load FAQ ({e}).")
        return faqs

    soup = BeautifulSoup(resp.text, "html.parser")
    main = soup.find("main", id="page-content")
    if not main:
        print("Warning: FAQ structure not found.")
        return faqs

    for qbox in main.select("div.question"):
        h3 = qbox.find("h3")
        ans = qbox.find("div", class_="answer")
        if not h3 or not ans:
            continue

        q_text = re.sub(r"\s+", " ", h3.get_text(" ", strip=True))
        a_text = re.sub(r"\s+", " ", ans.get_text(" ", strip=True))
        faqs.append(
            Document(
                page_content=f"Frage: {q_text}\nAntwort: {a_text}",
                metadata={"source": url, "type": "faq", "question": q_text},
            )
        )

    print(f"FAQ entries loaded: {len(faqs)}")
    return faqs


def load_pdf(path: str, label: str) -> List[Document]:
    """
    Load a PDF page by page and attach consistent metadata.
    Returns an empty list if the file is missing or cannot be read.
    """
    if not os.path.exists(path):
        print(f"Warning: File not found: {path}")
        return []

    try:
        loader = PyPDFLoader(path)
        pages = loader.load()
        for d in pages:
            d.metadata.update({"type": "pdf", "pdf_label": label, "source": path})
        print(f"{label} pages loaded: {len(pages)}")
        return pages
    except Exception as e:
        print(f"Warning: Could not load PDF ({path}): {e}")
        return []


def load_all_sources() -> List[Document]:
    """Combine FAQ entries and all PDF pages into a single document list."""
    faqs = scrape_faq()
    pdf_docs: List[Document] = []
    for path, label in PDFS:
        pdf_docs.extend(load_pdf(path, label))
    docs = faqs + pdf_docs
    print(f"Total documents loaded: {len(docs)}")
    return docs


def unique_sources(docs: List[Document], max_items: int = 3) -> List[Tuple[str, str]]:
    """
    Build a deduplicated source list, preserving insertion order.
    PDFs are identified by filename; URLs are normalized by stripping trailing slashes.
    """
    seen = set()
    out = []
    for d in docs:
        src = d.metadata.get("source", "")
        if not src:
            continue
        key = os.path.basename(src) if src.lower().endswith(".pdf") else src.rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        label = d.metadata.get("pdf_label") or d.metadata.get("type") or ""
        out.append((src, label))
        if len(out) >= max_items:
            break
    return out


def dedupe_docs(docs: List[Document]) -> List[Document]:
    """
    Remove duplicate or near-identical documents from a retrieval result
    so the same FAQ entry or PDF chunk does not appear multiple times in the context.
    """
    seen = set()
    out: List[Document] = []

    for d in docs:
        src = d.metadata.get("source", "")
        page = d.metadata.get("page", "")
        question = d.metadata.get("question", "")
        content = re.sub(r"\s+", " ", d.page_content).strip()

        key = (
            src.rstrip("/") if isinstance(src, str) else src,
            page,
            question,
            content[:300].lower(),
        )

        if key in seen:
            continue
        seen.add(key)
        out.append(d)

    return out


# ---------------------------------------------------------------------------
# Text splitting and vector store
# ---------------------------------------------------------------------------

def split_documents(docs: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks for more stable retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(docs)
    print(f"Text chunks created: {len(chunks)}")
    return chunks


def build_vectorstore(split_docs: List[Document]) -> Chroma:
    """
    Build a persistent Chroma vector store using OpenAI embeddings.
    If RESET_VECTORSTORE_ON_START is True, any existing store is deleted first
    to prevent duplicates from previous runs.
    """
    emb = OpenAIEmbeddings(model=EMBED_MODEL)

    if RESET_VECTORSTORE_ON_START and os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)

    vs = Chroma.from_documents(
        documents=split_docs,
        embedding=emb,
        persist_directory=PERSIST_DIR,
    )

    try:
        print(f"Vectors stored: {vs._collection.count()}")
    except Exception:
        pass

    return vs


# ---------------------------------------------------------------------------
# RAG chain: retriever + prompt + LLM + parser
# ---------------------------------------------------------------------------

def format_docs(docs: List[Document]) -> str:
    """Join multiple document contents into a single context string for the LLM."""
    return "\n".join(doc.page_content for doc in docs)


def make_context_preview(context: str, max_chars: int = 700) -> str:
    """Return a short, readable preview of the retrieved context (for debug output)."""
    if not context:
        return ""
    compact = re.sub(r"\s+", " ", context).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rstrip() + " ..."


def format_chat_history(chat_history: List[Dict[str, Any]]) -> str:
    """
    Format the conversation history as a plain text block for the prompt.
    Expected message format: {"role": "user"|"assistant", "content": "..."}
    """
    if not chat_history:
        return "No previous conversation."

    lines = []
    for msg in chat_history:
        role = msg.get("role", "")
        content = (msg.get("content", "") or "").strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")

    return "\n".join(lines) if lines else "No previous conversation."


def reformulate_question(
    llm,
    question: str,
    chat_history: List[Dict[str, Any]],
) -> str:
    """
    Rewrite a follow-up question into a standalone search query using the
    conversation history, so retrieval stays reliable even for short follow-ups.
    If there is no history, the original question is returned unchanged.
    """
    history_text = format_chat_history(chat_history)

    if history_text == "No previous conversation.":
        return question

    chain = HISTORY_AWARE_RETRIEVAL_PROMPT | llm | StrOutputParser()
    rewritten = chain.invoke({"chat_history": history_text, "input": question})
    rewritten = (rewritten or "").strip()
    return rewritten if rewritten else question


def build_rag_chain(vectorstore: Chroma) -> dict:
    """
    Set up the answer pipeline:
      - retriever: used separately for context retrieval
      - prompt: template with context and conversation history
      - LLM: OpenAI chat model
      - parser: converts model output to a plain string
    Returns a dict with 'answer_chain' and 'reformulation_llm'.
    """
    retriever = vectorstore.as_retriever()

    global RETRIEVER
    RETRIEVER = retriever

    answer_llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)
    reformulation_llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)

    rag_chain = VIENNABASE_RAG_PROMPT | answer_llm | StrOutputParser()
    return {
        "answer_chain": rag_chain,
        "reformulation_llm": reformulation_llm,
    }


# ---------------------------------------------------------------------------
# Context retrieval
# ---------------------------------------------------------------------------

def retrieve_context(
    retriever,
    query: str,
    k: int = 4,
    vectorstore=None,
    max_distance: float = EVIDENCE_DISTANCE_THRESHOLD,
) -> Tuple[str, List[dict]]:
    """
    Retrieve relevant context and a deduplicated source list for a given query.
    Uses distance-based filtering when a vector store is available.
    Falls back to plain retriever if scoring fails.
    """
    if not USE_EVIDENCE_THRESHOLD:
        max_distance = float("inf")

    docs = []
    if vectorstore is not None:
        try:
            scored = vectorstore.similarity_search_with_score(query, k=k)
            docs = [
                d for (d, dist) in scored
                if d is not None
                and dist is not None
                and dist <= max_distance
                and d.page_content
                and len(d.page_content.strip()) > 50
            ]
        except Exception:
            docs = retriever.invoke(query)
    else:
        docs = retriever.invoke(query)

    docs = dedupe_docs(docs)

    if not docs:
        return "", []

    context = format_docs(docs)
    pairs = unique_sources(docs, max_items=min(k, 3))
    sources = [{"source": src, "label": label} for (src, label) in pairs]
    return context, sources


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------

def ask(rag_chain: dict, query: str) -> None:
    """
    Process a single user query through the full RAG pipeline:
      1. Validate input
      2. Retrieve evidence (with distance filtering if available)
      3. Generate answer
      4. Print answer and sources
    """
    try:
        query = validate_query(query)
    except ValueError as e:
        print(f"Warning: {e}")
        return

    try:
        docs = RETRIEVER.invoke(query)
    except Exception:
        docs = []

    filtered_docs = None
    try:
        scored = RETRIEVER.vectorstore.similarity_search_with_score(query, k=4)
        if scored:
            filtered_docs = [
                d for (d, dist) in scored
                if d is not None
                and dist is not None
                and dist <= EVIDENCE_DISTANCE_THRESHOLD
                and d.page_content
                and len(d.page_content.strip()) > 50
            ]
    except Exception:
        pass

    if not docs:
        print("No relevant context found for this question in the available documents.")
        print("Try rephrasing with specific keywords (e.g. from AGBs, Heimstatut, or Aufnahmerichtlinien).")
        return

    if filtered_docs is not None:
        sources = [] if not filtered_docs else unique_sources(filtered_docs, max_items=3)
    else:
        sources = unique_sources(docs, max_items=3)

    context_text = format_docs(
        filtered_docs if filtered_docs is not None and filtered_docs else docs
    )

    answer = rag_chain["answer_chain"].invoke({
        "context": context_text,
        "chat_history": "No previous conversation.",
        "input": query,
    })

    if not is_safe_text(answer):
        print("Warning: The generated answer was blocked by the moderation check.")
        return

    print(answer)

    if sources:
        print("\nSources:")
        seen = set()
        for item in sources:
            if item in seen:
                continue
            seen.add(item)
            src, label = item
            suffix = f" [{label}]" if label else ""
            print(f"  - {src}{suffix}")


# ---------------------------------------------------------------------------
# Entry point (CLI loop)
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Run the full pipeline:
      - Load environment and documents
      - Build vector store and RAG chain
      - Start an interactive CLI chat loop
    """
    load_env()
    docs = load_all_sources()
    if not docs:
        print("No documents loaded — aborting.")
        return

    split_docs = split_documents(docs)
    if not split_docs:
        print("No text chunks produced — aborting.")
        return

    vectorstore = build_vectorstore(split_docs)
    rag_chain = build_rag_chain(vectorstore)

    print("\nChat mode started (type 'exit' to quit):")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if q.lower() in {"exit", "quit"}:
            print("Exiting.")
            break
        ask(rag_chain, q)


if __name__ == "__main__":
    main()