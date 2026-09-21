import os
import time

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "document.pdf")
DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "documento_pdf")
PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
BATCH_SIZE = 10
MAX_RETRIES = 5
RETRY_WAIT_SECONDS = 30


def get_embeddings():
    """Retorna o modelo de embeddings conforme o provedor configurado no .env."""
    if PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")
        return GoogleGenerativeAIEmbeddings(model=model)

    from langchain_openai import OpenAIEmbeddings

    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model)


def ingest_pdf():
    if not DATABASE_URL:
        raise SystemExit("DATABASE_URL não definida. Copie o .env.example para .env e preencha.")

    if not os.path.exists(PDF_PATH):
        raise SystemExit(f"PDF não encontrado em '{PDF_PATH}'. Verifique a variável PDF_PATH.")

    print(f"Carregando PDF: {PDF_PATH}")
    docs = PyPDFLoader(PDF_PATH).load()
    print(f"{len(docs)} página(s) carregada(s).")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=False,
    )
    chunks = splitter.split_documents(docs)
    print(f"{len(chunks)} chunk(s) gerado(s) (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

    embeddings = get_embeddings()

    store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )

    print(f"Gravando embeddings na collection '{COLLECTION_NAME}'...")
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                store.add_documents(batch)
                break
            except Exception as exc:
                if attempt == MAX_RETRIES:
                    raise
                print(f"Falha no lote {start // BATCH_SIZE + 1} ({exc.__class__.__name__}); "
                      f"nova tentativa em {RETRY_WAIT_SECONDS}s ({attempt}/{MAX_RETRIES})...")
                time.sleep(RETRY_WAIT_SECONDS)
        print(f"  {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks gravados")
    print("Ingestão concluída com sucesso.")


if __name__ == "__main__":
    ingest_pdf()
