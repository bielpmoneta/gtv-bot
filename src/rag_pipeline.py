"""
Pipeline de RAG do GTV.

Fluxo:

Pergunta
    ↓
Busca semântica FAISS
    ↓
Busca textual
    ↓
Combinação dos resultados
    ↓
LLM
    ↓
Resposta / NAO_SEI
"""

import re

from config.config import OPENAI_API_KEY

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_INDICE = "data/faiss_index"

NOME_MODELO_EMBEDDING = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

MARCADOR_NAO_SEI = "NAO_SEI"

# Quantos documentos o FAISS procura inicialmente.
K_BUSCA_SEMANTICA = 10

# Quantos documentos chegam ao LLM.
K_DOCUMENTOS_FINAIS = 5


# ============================================================
# EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name=NOME_MODELO_EMBEDDING
)


# ============================================================
# FAISS
# ============================================================

vectorstore = FAISS.load_local(
    PASTA_INDICE,
    embeddings,
    allow_dangerous_deserialization=True,
)


# ============================================================
# DOCUMENTOS
# ============================================================

# Recupera os documentos armazenados no índice.
#
# Isso permite fazer uma busca textual complementar à busca
# semântica do FAISS.

todos_documentos = list(
    vectorstore.docstore._dict.values()
)


# ============================================================
# NORMALIZAÇÃO
# ============================================================

def normalizar_texto(texto: str) -> str:
    """
    Normaliza o texto para facilitar comparação de palavras.
    """

    texto = texto.lower()

    texto = re.sub(
        r"[^\w\s]",
        " ",
        texto,
        flags=re.UNICODE,
    )

    texto = re.sub(
        r"\s+",
        " ",
        texto,
    )

    return texto.strip()


# ============================================================
# BUSCA TEXTUAL
# ============================================================

def busca_textual(
    pergunta: str,
    documentos: list[Document],
    limite: int = 10,
) -> list[Document]:

    pergunta_normalizada = normalizar_texto(
        pergunta
    )

    palavras = {
        palavra
        for palavra in pergunta_normalizada.split()
        if len(palavra) >= 3
    }

    resultados = []

    for documento in documentos:

        texto = normalizar_texto(
            documento.page_content
        )

        pontuacao = 0

        for palavra in palavras:

            if palavra in texto:
                pontuacao += 1

        if pontuacao > 0:

            resultados.append(
                (
                    pontuacao,
                    documento,
                )
            )

    resultados.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        documento
        for _, documento in resultados[:limite]
    ]


# ============================================================
# BUSCA HÍBRIDA
# ============================================================

def buscar_documentos(
    pergunta: str,
) -> list[Document]:

    documentos_finais = []

    # --------------------------------------------------------
    # BUSCA SEMÂNTICA
    # --------------------------------------------------------

    resultados_semanticos = (
        vectorstore.similarity_search(
            pergunta,
            k=K_BUSCA_SEMANTICA,
        )
    )

    # --------------------------------------------------------
    # BUSCA TEXTUAL
    # --------------------------------------------------------

    resultados_textuais = busca_textual(
        pergunta,
        todos_documentos,
        limite=K_BUSCA_SEMANTICA,
    )

    # --------------------------------------------------------
    # COMBINA RESULTADOS
    # --------------------------------------------------------

    vistos = set()

    resultados_combinados = (
        resultados_semanticos
        + resultados_textuais
    )

    for documento in resultados_combinados:

        fonte = documento.metadata.get(
            "source",
            "",
        )

        pagina = documento.metadata.get(
            "page",
            "",
        )

        chave = (
            fonte,
            pagina,
            documento.page_content[:100],
        )

        if chave in vistos:
            continue

        vistos.add(chave)

        documentos_finais.append(
            documento
        )

        if len(documentos_finais) >= K_DOCUMENTOS_FINAIS:
            break

    return documentos_finais


# ============================================================
# LLM
# ============================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPENAI_API_KEY,
)


# ============================================================
# PROMPT
# ============================================================

system_prompt = f"""
Você é um assistente especializado no sistema GTV.

Sua função é responder dúvidas utilizando exclusivamente
as informações presentes nos documentos fornecidos.

REGRAS:

1. Use SOMENTE o contexto fornecido.

2. Não invente informações.

3. Não utilize conhecimento externo.

4. Se o contexto possuir informação suficiente para responder,
responda normalmente.

5. Se não houver informação suficiente, responda exatamente:

{MARCADOR_NAO_SEI}

6. Não retorne {MARCADOR_NAO_SEI} quando for possível responder
com segurança utilizando parte das informações presentes.

7. Para perguntas sobre procedimentos, apresente as etapas
descritas nos documentos.

8. Seja direto e claro.

9. Não mencione o contexto, FAISS, RAG, embeddings ou documentos
internos.

10. Não ultrapasse 5 frases, salvo quando uma lista de etapas
for necessária.

CONTEXTO:

{{context}}
"""


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt,
        ),
        (
            "human",
            "{input}",
        ),
    ]
)


# ============================================================
# CHAIN
# ============================================================

question_answer_chain = (
    create_stuff_documents_chain(
        llm,
        prompt,
    )
)


# ============================================================
# RESPOSTA
# ============================================================

def answer_question(
    question: str,
) -> str:
    """
    Busca documentos relevantes e gera a resposta.
    """

    documentos = buscar_documentos(
        question
    )

    if not documentos:
        return MARCADOR_NAO_SEI

    resultado = question_answer_chain.invoke(
        {
            "input": question,
            "context": documentos,
        }
    )

    return resultado.strip()


# ============================================================
# ATENDIMENTO HUMANO
# ============================================================

def precisa_de_atendente_humano(
    resposta: str,
) -> bool:

    return (
        resposta.strip().upper()
        == MARCADOR_NAO_SEI
    )