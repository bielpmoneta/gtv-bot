"""
Teste manual/diagnóstico do pipeline RAG contra o índice FAISS REAL.

Diferente de test_rag_pipeline.py (que mocka tudo), este script usa:
- o índice FAISS de verdade (data/faiss_index)
- o modelo de embeddings de verdade
- a API da OpenAI de verdade (gasta tokens!)

Serve pra avaliar QUALIDADE de retrieval e de resposta, não corretude
de código. Rode isso depois que os testes automatizados já passarem.

Uso:
    python test_faiss_manual.py            -> modo interativo
    python test_faiss_manual.py --lote     -> roda a lista de casos abaixo
"""

import os
import sys
from pathlib import Path

# C:\gtv-bot\src (onde este script e rag_pipeline.py estão)
PASTA_SRC = Path(__file__).resolve().parent
# C:\gtv-bot (raiz do projeto, onde ficam config/ e data/)
PASTA_RAIZ = PASTA_SRC.parent

sys.path.insert(0, str(PASTA_SRC))
sys.path.insert(0, str(PASTA_RAIZ))

# rag_pipeline.py usa "data/faiss_index" como caminho relativo ao
# diretório de trabalho, não ao arquivo — então precisamos rodar
# a partir da raiz do projeto para o índice ser encontrado.
os.chdir(PASTA_RAIZ)

import rag_pipeline as rp


# ============================================================
# CASOS DE TESTE EM LOTE
# ============================================================
# Edite esta lista com perguntas reais do domínio do GTV.
# "esperado": None -> só reporta a resposta, sem validar
# "esperado": "NAO_SEI" -> valida que o pipeline recusa (assunto fora do escopo)
# "esperado": "algo" -> valida que a resposta contém esse trecho (substring, case-insensitive)

CASOS_DE_TESTE = [
    {
        "pergunta": "Como cadastrar um novo chamamento?",
        "esperado": "algo",
    },
    {
        "pergunta": "Como excluir um chamamento?",
        "esperado": "algo",
    },
    {
        "pergunta": "Como faço para trocar meu carro por um avião?",
        "esperado": "NAO_SEI",
    },
    # Adicione mais casos aqui, com perguntas reais do seu domínio
    # e trechos que você sabe que devem aparecer na resposta.
]


# ============================================================
# DIAGNÓSTICO DE UMA PERGUNTA
# ============================================================

def diagnosticar(pergunta: str) -> None:
    """
    Mostra o que a busca semântica e textual estão trazendo
    para uma pergunta, e a resposta final do pipeline.
    """

    print("=" * 70)
    print(f"PERGUNTA: {pergunta}")
    print("=" * 70)

    # --- busca semântica com score ---
    print("\n[BUSCA SEMÂNTICA] (top {})".format(rp.K_BUSCA_SEMANTICA))
    resultados_score = rp.vectorstore.similarity_search_with_score(
        pergunta,
        k=rp.K_BUSCA_SEMANTICA,
    )
    for i, (doc, score) in enumerate(resultados_score, start=1):
        fonte = doc.metadata.get("source", "?")
        pagina = doc.metadata.get("page", "?")
        trecho = doc.page_content[:100].replace("\n", " ")
        print(f"  {i}. score={score:.4f}  [{fonte} p{pagina}]  {trecho}...")

    # --- busca textual ---
    print("\n[BUSCA TEXTUAL] (top {})".format(rp.K_BUSCA_SEMANTICA))
    resultados_textuais = rp.busca_textual(
        pergunta,
        rp.todos_documentos,
        limite=rp.K_BUSCA_SEMANTICA,
    )
    if not resultados_textuais:
        print("  (nenhum resultado)")
    for i, doc in enumerate(resultados_textuais, start=1):
        fonte = doc.metadata.get("source", "?")
        pagina = doc.metadata.get("page", "?")
        trecho = doc.page_content[:100].replace("\n", " ")
        print(f"  {i}. [{fonte} p{pagina}]  {trecho}...")

    # --- documentos finais combinados ---
    documentos_finais = rp.buscar_documentos(pergunta)
    print(f"\n[DOCUMENTOS FINAIS ENVIADOS AO LLM] ({len(documentos_finais)})")
    for i, doc in enumerate(documentos_finais, start=1):
        fonte = doc.metadata.get("source", "?")
        pagina = doc.metadata.get("page", "?")
        print(f"  {i}. [{fonte} p{pagina}]")

    # --- resposta final ---
    print("\n[RESPOSTA]")
    resposta = rp.answer_question(pergunta)
    print(f"  {resposta}")

    if rp.precisa_de_atendente_humano(resposta):
        print("\n  -> Acionaria atendimento humano (NAO_SEI)")

    print()


# ============================================================
# MODO INTERATIVO
# ============================================================

def modo_interativo() -> None:
    print(f"Índice carregado com {len(rp.todos_documentos)} documentos.")
    print("Digite uma pergunta (ou 'sair' para encerrar).\n")

    while True:
        pergunta = input("Pergunta> ").strip()

        if not pergunta:
            continue

        if pergunta.lower() in ("sair", "exit", "quit"):
            break

        diagnosticar(pergunta)


# ============================================================
# MODO LOTE
# ============================================================

def modo_lote() -> None:
    total = len(CASOS_DE_TESTE)
    acertos = 0
    falhas = []

    for caso in CASOS_DE_TESTE:
        pergunta = caso["pergunta"]
        esperado = caso["esperado"]

        resposta = rp.answer_question(pergunta)

        if esperado is None:
            status = "INFO"
            acertos += 1  # não validado, não conta como falha
        elif esperado.upper() == "NAO_SEI":
            ok = rp.precisa_de_atendente_humano(resposta)
            status = "OK" if ok else "FALHOU"
            if ok:
                acertos += 1
            else:
                falhas.append(caso)
        else:
            ok = esperado.lower() in resposta.lower()
            status = "OK" if ok else "FALHOU"
            if ok:
                acertos += 1
            else:
                falhas.append(caso)

        print(f"[{status}] {pergunta}")
        print(f"       -> {resposta}\n")

    print("=" * 70)
    print(f"Resultado: {acertos}/{total}")

    if falhas:
        print("\nCasos que falharam:")
        for caso in falhas:
            print(f"  - {caso['pergunta']} (esperado: {caso['esperado']})")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    if "--lote" in sys.argv:
        modo_lote()
    else:
        modo_interativo()