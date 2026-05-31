# Viennabase RAG Chatbot

**Masterarbeit — FH Technikum Wien**

- **Autor:** Ridha Omrane, BSc
- **Studiengang:** AI Engineering (MSc)
- **Betreuer:** Dipl.-Ing. Dipl.-Ing. Dr. techn. Christoph Redl BSc
- **Co-Betreuer:** Mag. Simon Kovacic

Bei der Entwicklung dieses Projekts wurde ChatGPT (GPT-5.4) zur Unterstützung bei der Implementierung verwendet sowie Claude (Opus 4.7) für Code-Cleaning, Kommentare und Dokumentation.

---

Retrieval-Augmented Generation (RAG) Chatbot für Viennabase-Dokumente (FAQs, AGBs, Aufnahmerichtlinien, Heimstatut).
Das System lädt die Inhalte lokal, erstellt einen Chroma-Vektorstore und beantwortet Fragen über eine FastAPI.
Ein Streamlit-Frontend bietet eine einfache Benutzeroberfläche für lokale Tests.
Zusätzlich ist der Chatbot über WhatsApp nutzbar.

## Funktionen

- **RAG-Pipeline**: semantische Suche (Chroma) + OpenAI-Chatmodell
- **Quellenangaben**: dedupliziert, maximal Top-3
- **Evidence-Check**: bei fehlender Evidenz werden keine Quellen angezeigt und eine feste Fallback-Antwort zurückgegeben
- **Guardrails**:
  - *Input*: Normalisierung, Längenlimit, Link-/E-Mail-Filter, Prompt-Injection-Heuristiken
  - *Output*: OpenAI-Moderation (fail-open bei Ausfall)
- **Frontend**: Streamlit-App als UI für die FastAPI
- **WhatsApp-Integration**: Empfang und Beantwortung von Textnachrichten über die WhatsApp Cloud API

## Architektur

```
chatbot.py           # RAG-Kernlogik (Laden, Splitten, Vectorstore, Chain, CLI)
api.py               # FastAPI: /ask, /health, WhatsApp-Webhook
streamlit_app.py     # Frontend (HTTP-Client zur FastAPI)
input_guardrails.py  # Eingabe-Validierung und Injection-Erkennung
output_guardrails.py # Ausgabe-Moderation (OpenAI)
prompts.py           # LangChain-Prompt-Templates
data/                # AGBs.pdf, Aufnahmerichtlinien.pdf, Heimstatut.pdf
chroma_db/           # Persistenter Vektorstore (wird beim Start erstellt)
```

## Voraussetzungen

- Python 3.12 (empfohlen)
- OpenAI API Key
- Für WhatsApp: Meta Developer App mit WhatsApp Cloud API und gültige Zugangsdaten

## Installation

1. Repository klonen und virtuelle Umgebung aktivieren.
2. Abhängigkeiten installieren:
   ```bash
   pip install -r requirements.txt
   ```
3. `.env` anhand von `.env.example` erstellen:
   ```bash
   cp .env.example .env
   ```
   `OPENAI_API_KEY` setzen. Optional: `APP_API_KEY` für geschützte API-Zugriffe.

   Für die WhatsApp-Integration zusätzlich:
   - `WHATSAPP_ACCESS_TOKEN`
   - `WHATSAPP_PHONE_NUMBER_ID`
   - `WHATSAPP_VERIFY_TOKEN`

## Backend starten (FastAPI)

```bash
uvicorn api:app --reload
```

- Healthcheck: `GET http://127.0.0.1:8000/health`
- Swagger UI: `http://127.0.0.1:8000/docs`
- Endpunkt: `POST /ask`
  - Header (optional): `x-api-key: <APP_API_KEY>`
  - Body: `{"question": "Wie groß sind die Zimmer?"}`
- WhatsApp-Webhook:
  - `GET /whatsapp/webhook` → Verifikation gegenüber Meta
  - `POST /whatsapp/webhook` → Eingehende WhatsApp-Nachrichten

### Beispiel (curl)

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -H "x-api-key: changeme" \
  -d '{"question": "Wie groß sind die Zimmer?"}'
```

## Frontend starten (Streamlit)

In einem separaten Terminal:

```bash
streamlit run streamlit_app.py
```

Die App greift standardmäßig auf `APP_API_URL` aus der `.env` zu; der Endpunkt ist auch in der Sidebar änderbar.

## WhatsApp-Integration

Für lokale Tests wird ngrok benötigt, um einen öffentlichen Webhook bereitzustellen.

Typischer Ablauf:
1. `uvicorn api:app --reload`
2. `ngrok http 8000`
3. Den ngrok-Link als Callback-URL für `/whatsapp/webhook` im Meta Developer Dashboard eintragen
4. Den Verify Token aus der `.env` im Dashboard hinterlegen

## Guardrails

**Input** (`input_guardrails.py`):
- Unicode-Normalisierung und Whitespace-Bereinigung
- Längenlimit (800 Zeichen)
- Keine Links oder E-Mail-Adressen erlaubt
- Heuristische Erkennung von Prompt-Injection-Mustern (DE/EN)

**Output** (`output_guardrails.py`):
- OpenAI-Moderation über `omni-moderation-latest`
- Fail-open: Bei API-Fehlern wird die Antwort nicht blockiert

## Persistenz

Der Vektorstore wird in `chroma_db/` gespeichert und beim Start der Anwendung neu aufgebaut.
Die WhatsApp-Chat-Historie wird nur im Arbeitsspeicher gehalten und geht bei einem Serverneustart verloren.

Um den Vektorstore manuell neu zu indizieren:
```bash
rm -rf chroma_db
```

## Troubleshooting

| Problem | Lösung |
|---|---|
| `OPENAI_API_KEY` fehlt | `.env` prüfen, Terminal neu starten |
| `401 Unauthorized` bei `/ask` | `APP_API_KEY` in `.env` und im Header prüfen |
| Quellen bei irrelevanten Fragen | `EVIDENCE_DISTANCE_THRESHOLD` in `chatbot.py` anpassen |
| WhatsApp antwortet nicht |