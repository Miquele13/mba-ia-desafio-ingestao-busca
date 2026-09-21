# Ingestão e Busca Semântica com LangChain e Postgres

Solução do desafio: ingestão de um PDF em PostgreSQL + pgVector e chat via CLI que responde **apenas** com base no conteúdo do documento.

## Como funciona

- **`src/ingest.py`** — carrega o `document.pdf` (PyPDFLoader), divide em chunks de **1000 caracteres com overlap de 150** (RecursiveCharacterTextSplitter), gera embeddings e grava os vetores no Postgres via `PGVector`.
- **`src/search.py`** — monta a chain: vetoriza a pergunta, busca os **10 chunks mais relevantes** (`similarity_search_with_score(query, k=10)`), concatena o contexto no prompt e chama a LLM.
- **`src/chat.py`** — loop de perguntas e respostas no terminal.

Perguntas fora do conteúdo do PDF retornam: `Não tenho informações necessárias para responder sua pergunta.`

## Pré-requisitos

- Python 3.10+
- Docker e Docker Compose
- API Key da OpenAI **ou** do Google Gemini

## Configuração

1. Clone o repositório e entre na pasta.

2. Crie e ative o ambiente virtual:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

4. Crie o arquivo de variáveis de ambiente:

```bash
cp .env.example .env
```

Edite o `.env` e preencha a chave do provedor escolhido:

- `LLM_PROVIDER=openai` → preencha `OPENAI_API_KEY`
- `LLM_PROVIDER=gemini` → preencha `GOOGLE_API_KEY`

`DATABASE_URL` já vem apontando para o banco do `docker-compose.yml`.

## Execução

1. Subir o banco de dados:

```bash
docker compose up -d
```

2. Executar a ingestão do PDF:

```bash
python src/ingest.py
```

3. Rodar o chat:

```bash
python src/chat.py
```

Exemplo:

```
PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

Digite `sair` para encerrar o chat.

## Troca de modelo de embeddings

A tabela de vetores é criada na primeira ingestão com a dimensão do modelo escolhido. Se trocar o modelo de embeddings depois, apague o volume do banco e refaça a ingestão:

```bash
docker compose down -v
docker compose up -d
python src/ingest.py
```
