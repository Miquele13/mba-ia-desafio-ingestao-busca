import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_postgres import PGVector

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "documento_pdf")
PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

TOP_K = 10

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def get_embeddings():
    """Retorna o modelo de embeddings conforme o provedor configurado no .env."""
    if PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")
        return GoogleGenerativeAIEmbeddings(model=model)

    from langchain_openai import OpenAIEmbeddings

    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model)


def get_llm():
    """Retorna o modelo de linguagem conforme o provedor configurado no .env."""
    if PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = os.getenv("GOOGLE_LLM_MODEL", "gemini-3.6-flash")
        return ChatGoogleGenerativeAI(model=model, temperature=0)

    from langchain_openai import ChatOpenAI

    model = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model, temperature=0)


def get_vector_store():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL não definida. Copie o .env.example para .env e preencha.")

    return PGVector(
        embeddings=get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )


def search_prompt(question=None):
    """Monta a chain de busca + resposta.

    Sem argumento, retorna a chain (usada pelo chat.py).
    Com uma pergunta, retorna diretamente a resposta.
    """
    try:
        store = get_vector_store()
        llm = get_llm()
    except Exception as exc:  # noqa: BLE001
        print(f"Erro ao inicializar a busca: {exc}")
        return None

    prompt = PromptTemplate(
        input_variables=["contexto", "pergunta"],
        template=PROMPT_TEMPLATE,
    )

    def montar_contexto(pergunta: str) -> dict:
        resultados = store.similarity_search_with_score(pergunta, k=TOP_K)
        contexto = "\n\n".join(doc.page_content for doc, _score in resultados)
        return {"contexto": contexto, "pergunta": pergunta}

    chain = RunnableLambda(montar_contexto) | prompt | llm | StrOutputParser()

    if question:
        return chain.invoke(question)

    return chain
