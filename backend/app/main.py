from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse

from app.api.chat import error_response, router as chat_router

PROJECT_NAME = "Ayurveda and Home Remedies Knowledge Bot"

app = FastAPI(title=PROJECT_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5176",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api")


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(request: Request, _error: RequestValidationError) -> JSONResponse:
    if request.url.path == "/api/chat":
        message = "Request must include a non-empty question no longer than 2000 characters."
    else:
        message = "Request contains invalid or missing fields."
    return error_response(message, 422)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok", "project": PROJECT_NAME}