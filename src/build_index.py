"""
Script de indexação dos documentos do GTV.

Execute:

    python -m src.build_index

Sempre execute novamente quando adicionar, remover ou alterar PDFs.
"""

import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_PDFS = "data/pdfs"
PASTA_INDICE = "data/faiss_index"

NOME_MODELO_EMBEDDING = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


# ============================================================
# CARREGAMENTO DOS PDFs
# ============================================================

def carregar_todos_os_pdfs(pasta: str):

    documentos = []

    if not os.path.exists(pasta):
        raise FileNotFoundError(
            f"A pasta '{pasta}' não existe."
        )

    arquivos_pdf = sorted(
        f
        for f in os.listdir(pasta)
        if f.lower().endswith(".pdf")
    )

    if not arquivos_pdf:
        raise FileNotFoundError(
            f"Nenhum PDF encontrado em '{pasta}'."
        )

    print()
    print("=" * 70)
    print("CARREGANDO DOCUMENTOS")
    print("=" * 70)

    for nome_arquivo in arquivos_pdf:

        caminho = os.path.join(
            pasta,
            nome_arquivo,
        )

        print(f"Carregando: {nome_arquivo}")

        loader = PyPDFLoader(caminho)

        paginas = loader.load()

        # Adicionamos informações úteis ao metadata
        for pagina in paginas:

            pagina.metadata["arquivo"] = nome_arquivo

            pagina.metadata["titulo_documento"] = os.path.splitext(
                nome_arquivo
            )[0]

        documentos.extend(paginas)

    return documentos


# ============================================================
# INDEXAÇÃO
# ============================================================

def construir_indice():

    documentos = carregar_todos_os_pdfs(
        PASTA_PDFS
    )

    print()
    print(f"Total de páginas carregadas: {len(documentos)}")

    # ========================================================
    # CHUNKING
    # ========================================================

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,

        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ", ",
            " ",
            "",
        ],
    )

    pedacos = text_splitter.split_documents(
        documentos
    )

    print(
        f"Total de chunks gerados: {len(pedacos)}"
    )

    # ========================================================
    # METADATA DOS CHUNKS
    # ========================================================

    for indice, chunk in enumerate(pedacos):

        chunk.metadata["chunk_id"] = indice

        # Garante que exista uma fonte
        if "source" not in chunk.metadata:
            chunk.metadata["source"] = (
                chunk.metadata.get("arquivo", "")
            )

    # ========================================================
    # EMBEDDINGS
    # ========================================================

    print()
    print("=" * 70)
    print("GERANDO EMBEDDINGS")
    print("=" * 70)

    embeddings = HuggingFaceEmbeddings(
        model_name=NOME_MODELO_EMBEDDING
    )

    # ========================================================
    # FAISS
    # ========================================================

    print("Criando índice FAISS...")

    vectorstore = FAISS.from_documents(
        documents=pedacos,
        embedding=embeddings,
    )

    # ========================================================
    # SALVAR
    # ========================================================

    os.makedirs(
        PASTA_INDICE,
        exist_ok=True,
    )

    vectorstore.save_local(
        PASTA_INDICE
    )

    print()
    print("=" * 70)
    print("INDEXAÇÃO CONCLUÍDA")
    print("=" * 70)

    print(
        f"PDFs processados: {len(os.listdir(PASTA_PDFS))}"
    )

    print(
        f"Páginas: {len(documentos)}"
    )

    print(
        f"Chunks: {len(pedacos)}"
    )

    print(
        f"Índice: {PASTA_INDICE}"
    )

    print("=" * 70)


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    construir_indice()