from pathlib import Path
from typing import List, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RAGEngine:
    """Recuperador simples de contexto baseado em TF-IDF."""

    def __init__(self, document_path: str):
        self.document_path = Path(document_path)
        self.text = self.document_path.read_text(encoding="utf-8")
        self.chunks = self._split_into_chunks(self.text)

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2)
        )
        self.matrix = self.vectorizer.fit_transform(self.chunks)

    @staticmethod
    def _split_into_chunks(text: str, max_chars: int = 450) -> List[str]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks, current = [], ""

        for paragraph in paragraphs:
            if len(current) + len(paragraph) + 2 <= max_chars:
                current = f"{current}\n\n{paragraph}".strip()
            else:
                if current:
                    chunks.append(current)
                current = paragraph

        if current:
            chunks.append(current)

        return chunks

    def retrieve(self, question: str, top_k: int = 3) -> List[Dict]:
        query_vector = self.vectorizer.transform([question])
        scores = cosine_similarity(query_vector, self.matrix)[0]
        ranked = scores.argsort()[::-1][:top_k]

        return [
            {"text": self.chunks[i], "score": float(scores[i])}
            for i in ranked
            if scores[i] > 0
        ]

    def build_prompt(self, question: str, contexts: List[Dict]) -> str:
        context = "\n\n---\n\n".join(item["text"] for item in contexts)

        return f"""Você é um assistente de suporte técnico.
Responda somente com base no contexto fornecido.
Se a informação não estiver no contexto, diga que não encontrou essa informação na base.

CONTEXTO:
{context}

PERGUNTA:
{question}

RESPOSTA:
"""


def create_engine() -> RAGEngine:
    return RAGEngine("data/base_conhecimento.txt")
