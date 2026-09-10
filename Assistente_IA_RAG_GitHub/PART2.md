# Assistente de IA com RAG

Projeto acadêmico/portfólio desenvolvido para demonstrar fundamentos de IA generativa, LLMs, Prompt Engineering, RAG, APIs REST, Git/GitHub e documentação técnica.

## Objetivo
Criar um assistente capaz de responder perguntas usando documentos internos como base de conhecimento. O fluxo é:

1. Receber uma pergunta.
2. Buscar trechos relevantes nos documentos.
3. Montar um prompt com o contexto recuperado.
4. Enviar o contexto para um LLM quando a API estiver configurada.
5. Retornar a resposta e as fontes utilizadas.

## Tecnologias
- Python 3.10+
- FastAPI
- scikit-learn (TF-IDF para recuperação local)
- OpenAI API (opcional, para geração com LLM)
- Uvicorn
- Git/GitHub

## Como executar

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
uvicorn src.api:app --reload
```

Acesse a documentação interativa em `http://127.0.0.1:8000/docs`.

Sem uma chave de API, o projeto continua funcionando em modo de recuperação e retorna os trechos mais relevantes. Para habilitar geração com LLM, configure `OPENAI_API_KEY` como variável de ambiente.

## Exemplo de requisição

POST `/ask`

```json
{
  "question": "Qual é o prazo para solicitar suporte?"
}
```

## Exemplo de resposta

```json
{
  "answer": "...",
  "sources": ["data/base_conhecimento.txt"]
}
```

## Estrutura

```text
assistente_ia_rag/
├── data/
│   └── base_conhecimento.txt
├── src/
│   ├── rag.py
│   └── api.py
├── requirements.txt
├── .gitignore
└── README.md
```

## O que este projeto demonstra
- Recuperação de contexto com TF-IDF.
- Construção de prompts com contexto.
- Integração opcional com LLM.
- Endpoint REST.
- Validação de entrada.
- Retorno de fontes.
- Documentação técnica.

> Projeto de portfólio/estudo. Não representa experiência profissional anterior.
