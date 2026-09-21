from search import search_prompt


def main():
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Chat iniciado. Digite sua pergunta (ou 'sair' para encerrar).\n")

    while True:
        try:
            pergunta = input("PERGUNTA: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando o chat.")
            break

        if not pergunta:
            continue

        if pergunta.lower() in {"sair", "exit", "quit"}:
            print("Encerrando o chat.")
            break

        try:
            resposta = chain.invoke(pergunta)
        except Exception as exc:  # noqa: BLE001
            print(f"Erro ao processar a pergunta: {exc}\n")
            continue

        print(f"RESPOSTA: {resposta}\n")


if __name__ == "__main__":
    main()
