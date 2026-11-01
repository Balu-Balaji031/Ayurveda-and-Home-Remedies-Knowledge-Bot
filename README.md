# Ayurveda and Home Remedies Knowledge Bot

React/TypeScript frontend and FastAPI backend for the Ayurveda and Home Remedies Knowledge Bot. The frontend uses Firebase Authentication and Cloud Firestore for account access and per-user chat history. The backend retrieves PDF passages from the existing ChromaDB collection and uses Gemini as its primary provider, with Groq as a fallback for temporary Gemini failures during grounded answers, remedy explanations, and follow-ups.

## Requirements

- Node.js and npm
- Python 3.10 or newer

## Frontend

In a terminal at the project root:

```powershell
cd frontend
npm install
npm run dev
```

Vite serves the frontend at `http://localhost:5173`.

## Backend

In a separate terminal at the project root:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The API health check is available at `http://localhost:8000/health` and returns:

```json
{
  "status": "ok",
  "project": "Ayurveda and Home Remedies Knowledge Bot"
}
```

## Environment configuration

Copy `frontend/.env.example` to `frontend/.env` and `backend/.env.example` to `backend/.env` when configuring local values. Frontend Firebase web app values belong in `frontend/.env`, which is ignored by Git. Configure `GEMINI_API_KEY` and `GEMINI_MODEL` in `backend/.env` for primary answer generation. Set `GROQ_API_KEY` and `GROQ_MODEL` there to enable fallback; the example model `openai/gpt-oss-20b` is listed in the [official Groq model documentation](https://console.groq.com/docs/models). Groq is called only for Gemini quota/rate-limit, listed server errors, timeout, or temporary connection failures; it is not used for authentication, configuration, or malformed-request errors. Provider credentials remain backend-only.

## Chat history security rules

Chat data is stored under `users/{userId}/conversations/{conversationId}`. The root `firestore.rules` file limits access to the signed-in owner and allows optional structured remedy/source metadata on messages so selections and follow-up context can be restored. Create/enable Cloud Firestore for the existing Firebase project, then deploy the rules from the project root:

```powershell
firebase deploy --only firestore:rules --project ayurveda-remedies-bot
```

Do not replace the rules with open test rules. The FastAPI backend is not involved in chat storage, and this frontend-only implementation does not add Firebase token verification to the API.

## Remedy chat

The frontend posts health concerns to `POST /api/recommendations`, selects remedies through `POST /api/remedies/explain`, and keeps selected-remedy context for `POST /api/chat/follow-up`. The existing `POST /api/chat` endpoint remains available for context-grounded free-form answers.

PDF-derived remedy text is accepted only when its displayed excerpts match retrieved passages; filenames and page numbers are retained with the conversation. When no relevant PDF remedy is found, the backend may return clearly labeled AI-generated supportive suggestions or abstain when neither configured provider can produce a safe suggestion. AI-generated suggestions are not attributed to the uploaded PDFs and are not a substitute for professional medical care.

PDF retrieval and source validation run once before generation; the same prompt and retrieved context are sent to Groq only when the primary Gemini request has a retryable failure. Groq documents strict JSON Schema output for `openai/gpt-oss-20b`, but strict mode requires every property while AyurBot's schema has optional fields. The fallback therefore requests JSON Object mode and retains AyurBot's existing parsing, excerpt, and source validation. For GPT-OSS models, the fallback sets `reasoning_effort=low` so hidden reasoning does not consume the full completion-token budget before producing the answer. No automatic provider retries are performed. The backend does not verify Firebase ID tokens. Firestore history is protected by owner-scoped rules, but API requests themselves are not authenticated.