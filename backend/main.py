"""
FastAPI backend for the medical RAG assistant.

Flow per request:
  query -> guardrails.check_query
        -> hybrid retrieval (vector + BM25)
        -> Gemini generation, grounded in retrieved chunks
        -> guardrails.append_disclaimer_if_needed
        -> log conversation + sources to DB
"""
import os
import time

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db.models import init_db, get_db, Conversation, Message, SourceCitation
from rag.retrieve import HybridRetriever
from rag.guardrails import check_query, append_disclaimer_if_needed

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = "gemini-2.5-flash"

app = FastAPI(title="Medical RAG Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever: HybridRetriever | None = None


@app.on_event("startup")
def startup():
    global retriever
    init_db()
    retriever = HybridRetriever()


SYSTEM_PROMPT = """You are a medical information assistant. Answer ONLY using
the provided context from WHO/CDC guidelines and medical references. If the
context doesn't contain the answer, say so explicitly rather than guessing.
Never state a personal diagnosis. Always be precise about what the source
material says vs. general knowledge."""


class ChatRequest(BaseModel):
    query: str
    conversation_id: str | None = None


class SourceOut(BaseModel):
    document_name: str
    page_number: int
    relevance_score: float
    excerpt: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    conversation_id: str
    guardrail_flagged: bool
    latency_ms: float


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, db: Session = Depends(get_db)):
    start = time.time()

    guardrail_result = check_query(req.query)

    conversation = None
    if req.conversation_id:
        conversation = db.get(Conversation, req.conversation_id)
    if not conversation:
        conversation = Conversation()
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

    db.add(Message(conversation_id=conversation.id, role="user", content=req.query))
    db.commit()

    if guardrail_result.block_generation:
        answer = guardrail_result.prepend_message
        sources = []
    else:
        chunks = retriever.retrieve(req.query)
        context = "\n\n".join(
            f"[Source: {c.source}, page {c.page_number}]\n{c.text}" for c in chunks
        )
        prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {req.query}"

        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        answer = append_disclaimer_if_needed(response.text, guardrail_result)
        sources = chunks

    latency_ms = (time.time() - start) * 1000

    assistant_msg = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=answer,
        guardrail_flagged=guardrail_result.flagged,
        guardrail_reason=guardrail_result.reason,
        latency_ms=latency_ms,
    )
    db.add(assistant_msg)
    db.commit()
    db.refresh(assistant_msg)

    for c in sources:
        db.add(SourceCitation(
            message_id=assistant_msg.id,
            document_name=c.source,
            page_number=str(c.page_number),
            chunk_text=c.text[:500],
            relevance_score=c.score,
        ))
    db.commit()

    return ChatResponse(
        answer=answer,
        sources=[
            SourceOut(
                document_name=c.source,
                page_number=c.page_number,
                relevance_score=c.score,
                excerpt=c.text[:300],
            ) for c in sources
        ],
        conversation_id=conversation.id,
        guardrail_flagged=guardrail_result.flagged,
        latency_ms=round(latency_ms, 1),
    )


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
