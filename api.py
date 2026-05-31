# api.py
# REST API for the Viennabase RAG chatbot, implemented with FastAPI.
# Acts as the interface between the RAG pipeline and the various clients.
#
# Endpoints:
#   GET  /health              — health check
#   POST /ask                 — direct API access (e.g. from Streamlit)
#   GET  /whatsapp/webhook    — webhook verification for Meta
#   POST /whatsapp/webhook    — incoming WhatsApp message handling

from fastapi import FastAPI, HTTPException, Header, Request, Query, Response
from pydantic import BaseModel, Field
import os
import requests
import chatbot
from chatbot import (
    load_env,
    load_all_sources,
    split_documents,
    build_vectorstore,
    build_rag_chain,
    retrieve_context,
    format_chat_history,
    reformulate_question,
    make_context_preview,
)
from input_guardrails import validate_query
from output_guardrails import is_safe_text


# Set to False to skip history-aware reformulation (ablation study: no_history config).
USE_HISTORY_REFORMULATION = True


# ---------------------------------------------------------------------------
# Language detection and fallback response
# ---------------------------------------------------------------------------

def detect_user_language(text: str) -> str:
    """
    Simple heuristic to distinguish German from English.
    Returns "de" for German, "en" for English (default).
    """
    t = text.lower()

    german_markers = [
        "wie", "was", "wo", "wann", "warum", "darf", "dürfen", "kann", "können",
        "ist", "sind", "ich", "mein", "meine", "muss", "müssen", "zimmer",
        "gäste", "gast", "adresse", "heim", "büro", "regel", "regeln",
        "übernachten", "kaution", "miete", "aufnahme", "studentenheim",
        "ä", "ö", "ü", "ß",
    ]

    for marker in german_markers:
        if marker in t:
            return "de"

    return "en"


def no_context_answer(question: str) -> str:
    """
    Return a fixed fallback message in the detected language of the question,
    used when no reliable evidence was found in the knowledge base.
    """
    lang = "de"

    if lang == "de":
        return (
            "Ich konnte in den verfügbaren Viennabase-Quellen keine verlässliche Antwort "
            "auf diese Frage finden. Bitte kontaktiere das Viennabase-Büro direkt oder "
            "frage die Heimleitung, um eine sichere Auskunft zu erhalten."
        )

    return (
        "I could not find reliable information for this question in the available "
        "Viennabase sources. Please contact the Viennabase office directly or ask the "
        "residence management to get a reliable answer."
    )


# ---------------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------------

app = FastAPI(title="Viennabase RAG API")

# In-memory chat history per WhatsApp phone number.
# Note: this is lost on server restart and is only suitable for a minimal setup.
WHATSAPP_CHAT_MEMORY: dict[str, list[dict]] = {}


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class AskBody(BaseModel):
    """Request body for POST /ask."""
    question: str
    chat_history: list[ChatMessage] = Field(default_factory=list)
    debug: bool = False


# ---------------------------------------------------------------------------
# Startup: initialize RAG pipeline
# ---------------------------------------------------------------------------

@app.on_event("startup")
def _startup():
    """
    Run once when the server starts.
    Loads documents, creates embeddings, and initializes the RAG chain.
    Results are stored in app.state so they can be reused across requests.
    """
    load_env()
    docs = load_all_sources()
    if not docs:
        raise RuntimeError("No documents loaded.")

    chunks = split_documents(docs)
    if not chunks:
        raise RuntimeError("No text chunks produced.")

    vs = build_vectorstore(chunks)
    chains = build_rag_chain(vs)
    app.state.answer_chain = chains["answer_chain"]
    app.state.reformulation_llm = chains["reformulation_llm"]
    app.state.retriever = chatbot.RETRIEVER
    app.state.vectorstore = vs


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Simple health check endpoint."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Core processing logic (shared by /ask and WhatsApp)
# ---------------------------------------------------------------------------

def process_question(question: str, chat_history: list[dict], debug: bool = False) -> dict:
    """
    Run the central chatbot logic for a given question and conversation history.

    This function is shared between the /ask endpoint and the WhatsApp webhook
    to avoid duplicating validation, retrieval, and generation logic.

    Args:
        question: The current user question.
        chat_history: Previous messages as [{"role": ..., "content": ...}, ...].
        debug: If True, additional debug fields are included in the response.

    Returns:
        A response dict with at least {"answer": str, "sources": list},
        plus optional debug fields if debug=True.
    """
    try:
        q = validate_query(question)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    history_text = format_chat_history(chat_history)

    if USE_HISTORY_REFORMULATION:
        retrieval_query = reformulate_question(
            llm=app.state.reformulation_llm,
            question=q,
            chat_history=chat_history,
        )
    else:
        retrieval_query = q

    context, sources = retrieve_context(
        app.state.retriever,
        retrieval_query,
        k=4,
        vectorstore=app.state.vectorstore,
        max_distance=chatbot.EVIDENCE_DISTANCE_THRESHOLD,
    )

    if not context:
        payload = {
            "answer": no_context_answer(q),
            "sources": [],
        }
        if debug:
            payload.update({
                "retrieval_query": retrieval_query,
                "debug_history_text": history_text,
                "debug_context_preview": "",
                "debug_no_context": True,
            })
        return payload

    answer = app.state.answer_chain.invoke({
        "context": context,
        "chat_history": history_text,
        "input": q,
    })

    if not is_safe_text(answer):
        raise HTTPException(
            status_code=400,
            detail="Answer was blocked by the moderation check."
        )

    payload = {"answer": answer}
    if sources:
        payload["sources"] = sources

    if debug:
        payload.update({
            "retrieval_query": retrieval_query,
            "debug_history_text": history_text,
            "debug_context_preview": make_context_preview(context),
            "debug_no_context": False,
        })

    return payload


# ---------------------------------------------------------------------------
# WhatsApp helpers
# ---------------------------------------------------------------------------

def send_whatsapp_text(to_number: str, message_text: str) -> None:
    """
    Send a plain text message via the WhatsApp Cloud API.
    Credentials (access token and phone number ID) are read from the environment.
    """
    access_token = os.getenv("WHATSAPP_ACCESS_TOKEN")
    phone_number_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID")

    if not access_token or not phone_number_id:
        raise RuntimeError("WHATSAPP_ACCESS_TOKEN or WHATSAPP_PHONE_NUMBER_ID is missing.")

    url = f"https://graph.facebook.com/v25.0/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": message_text[:4096]},
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()


# ---------------------------------------------------------------------------
# /ask endpoint
# ---------------------------------------------------------------------------

@app.post("/ask")
def ask_api(body: AskBody, x_api_key: str = Header(default="")):
    """
    Direct API endpoint, primarily used by the Streamlit frontend.
    Validates the API key if one is configured, then delegates to process_question().
    """
    expected = os.getenv("APP_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

    history_payload = [msg.model_dump() for msg in body.chat_history]
    return process_question(
        question=body.question,
        chat_history=history_payload,
        debug=body.debug,
    )


# ---------------------------------------------------------------------------
# WhatsApp webhook
# ---------------------------------------------------------------------------

@app.get("/whatsapp/webhook")
def verify_whatsapp_webhook(
    hub_mode: str = Query(alias="hub.mode"),
    hub_verify_token: str = Query(alias="hub.verify_token"),
    hub_challenge: str = Query(alias="hub.challenge"),
):
    """
    Verify the WhatsApp webhook with Meta.
    Meta sends a GET request with hub.mode, hub.verify_token, and hub.challenge
    when the webhook is first configured; we return hub.challenge on success.
    """
    expected = os.getenv("WHATSAPP_VERIFY_TOKEN", "")
    if hub_mode == "subscribe" and hub_verify_token == expected:
        return Response(content=hub_challenge, media_type="text/plain")

    raise HTTPException(status_code=403, detail="Webhook verification failed.")


@app.post("/whatsapp/webhook")
async def whatsapp_webhook(request: Request):
    """
    Receive incoming WhatsApp messages and respond using the RAG pipeline.
    Only text messages are handled; status events and other types are ignored.
    """
    try:
        body = await request.json()

        entry = body["entry"][0]
        change = entry["changes"][0]
        value = change["value"]

        if "messages" not in value:
            return {"status": "ignored"}

        message = value["messages"][0]

        if message.get("type") != "text":
            return {"status": "ignored_non_text"}

        user_number = message["from"]
        user_text = message["text"]["body"]

        history = WHATSAPP_CHAT_MEMORY.get(user_number, [])

        result = process_question(
            question=user_text,
            chat_history=history,
            debug=False,
        )

        answer_text = result.get(
            "answer",
            "Entschuldigung, ich konnte gerade keine Antwort erzeugen."
        )

        send_whatsapp_text(user_number, answer_text)

        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": answer_text})
        WHATSAPP_CHAT_MEMORY[user_number] = history[-10:]

        return {"status": "ok"}

    except HTTPException as e:
        try:
            body = await request.json()
            value = body["entry"][0]["changes"][0]["value"]
            if "messages" in value:
                msg = value["messages"][0]
                user_number = msg.get("from")
                if user_number:
                    send_whatsapp_text(
                        user_number,
                        "Deine Nachricht konnte leider nicht verarbeitet werden."
                    )
        except Exception:
            pass
        raise e

    except Exception as e:
        print("WhatsApp webhook error:", e)
        return {"status": "error", "detail": str(e)}
