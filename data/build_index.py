"""
Script de INDEXAÇÃO dos documentos do GTV.

Isso roda SEPARADO da aplicação principal, e só precisa ser executado:
- uma vez, no início
- de novo, sempre que você adicionar/remover/atualizar PDFs

Gerar embeddings é a parte "cara" (em tempo de processamento) do RAG —
por isso ela fica isolada aqui, e não dentro do fluxo de pergunta/resposta.

Uso:
    python -m src.build_index
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

PASTA_PDFS = "data/pdfs"
PASTA_INDICE = "data/faiss_index"

# Modelo de embeddings LOCAL e GRATUITO (roda na sua máquina, sem custo
# por token). Esse modelo específico foi treinado para várias línguas,
# incluindo português, e é leve o suficiente para rodar sem GPU.
NOME_MODELO_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def carregar_todos_os_pdfs(pasta: str) -> list:
    """
    Percorre a pasta e carrega TODOS os arquivos .pdf encontrados.
    Cada página de cada PDF vira um "documento" separado no LangChain,
    com metadata indicando de qual arquivo/página ela veio (útil para
    depuração e, futuramente, para citar a fonte na resposta).
    """
    documentos = []
    arquivos_pdf = [f for f in os.listdir(pasta) if f.lower().endswith(".pdf")]

    if not arquivos_pdf:
        raise FileNotFoundError(
            f"Nenhum PDF encontrado em '{pasta}'. Coloque os documentos do GTV lá antes de indexar."
        )

    for nome_arquivo in arquivos_pdf:
        caminho = os.path.join(pasta, nome_arquivo)
        print(f"Carregando: {caminho}")
        loader = PyPDFLoader(caminho)
        documentos.extend(loader.load())

    return documentos


def construir_indice():
    documentos = carregar_todos_os_pdfs(PASTA_PDFS)
    print(f"Total de páginas carregadas: {len(documentos)}")

    # Quebra os documentos em pedaços menores (chunks). Pedaços menores
    # custam menos tokens quando forem enviados pro LLM depois, já que
    # só os pedaços mais relevantes de cada pergunta são usados
    # (não o documento inteiro).
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    pedacos = text_splitter.split_documents(documentos)
    print(f"Total de pedaços (chunks) gerados: {len(pedacos)}")

    print("Gerando embeddings localmente (a primeira execução baixa o modelo, ~120MB)...")
    embeddings = HuggingFaceEmbeddings(model_name=NOME_MODELO_EMBEDDING)

    vectorstore = FAISS.from_documents(documents=pedacos, embedding=embeddings)

    os.makedirs(PASTA_INDICE, exist_ok=True)
    vectorstore.save_local(PASTA_INDICE)
    print(f"Índice salvo em: {PASTA_INDICE}")
    print("Pronto! Agora o rag_pipeline.py pode carregar esse índice sem reprocessar os PDFs.")


if __name__ == "__main__":
    construir_indice()