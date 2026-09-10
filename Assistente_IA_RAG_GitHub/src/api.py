import os
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.rag import create_engine

app = FastAPI(
    title="Assistente de IA com RAG",
    version="1.0.0",
    description="API REST de portfólio para recuperação de contexto e geração opcional com LLM."
)

engine = create_engine()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(default=3, ge=1, le=5)


class AskResponse(BaseModel):
    answer: str
    sources: List[str]


def generate_with_llm(prompt: str) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        input=prompt
    )
    return response.output_text


@app.get("/")
def health_check():
    return {"status": "online", "project": "Assistente de IA com RAG"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    contexts = engine.retrieve(request.question, request.top_k)

    if not contexts:
        return AskResponse(
            answer="Não encontrei informações relevantes na base de conhecimento.",
            sources=[]
        )

    prompt = engine.build_prompt(request.question, contexts)
    llm_answer = generate_with_llm(prompt)

    if llm_answer:
        answer = llm_answer
    else:
        answer = (
            "Modo local: encontrei os seguintes trechos relevantes na base de conhecimento:\n\n"
            + "\n\n---\n\n".join(item["text"] for item in contexts)
        )

    return AskResponse(
        answer=answer,
        sources=["data/base_conhecimento.txt"]
    )
