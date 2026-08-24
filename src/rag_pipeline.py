import os
import re
import unicodedata

from config.config import OPENAI_API_KEY
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# ============================================================
# CONFIG
# ============================================================

PASTA_INDICE = "data/faiss_index"
MODELO_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

MARCADOR_NAO_SEI = "NAO_SEI"

K_SEMANTICA = 15
K_TEXTUAL = 15
K_FINAL = 5

PESO_SEMANTICO = 1.0
PESO_TEXTUAL = 1.5
BOOST_TITULO = 8.0
BOOST_CONTEUDO = 0.3
BOOST_INTENCAO_TITULO = 15.0

STOPWORDS = {
    "a", "à", "ao", "aos", "as", "com", "como", "da", "das", "de", "do", "dos", "e", "em", "essa", "esse", "esta", "este", "eu", "fazer",
    "faço", "faco", "foi", "for", "há", "isso", "isto", "na", "nas", "no", "nos", "o", "os", "ou", "para", "por", "que", "se", "sem", "ser",
    "um", "uma", "umas", "uns", "uso", "faco", "fazer", "consigo", "posso", "preciso", "quero",
}

EQUIVALENCIAS = {
    "cadastrar": "cadastro",
    "cadastramento": "cadastro",
    "cadastrada": "cadastro",
    "cadastrado": "cadastro",
    "cadastros": "cadastro",
    "entes": "ente",
    "usuarios": "usuario",
    "usuaria": "usuario",
    "usuarias": "usuario",
    "convenentes": "convenente",
    "chamamentos": "chamamento",
    "propostas": "proposta",
    "anexos": "anexo",
    "excluir": "exclusao",
    "exclusao": "exclusao",
    "ativar": "ativacao",
    "ativacao": "ativacao",
    "inativar": "inativacao",
    "inativacao": "inativacao",
    "consultar": "consulta",
    "consulta": "consulta",
    "detalhar": "detalhe",
    "detalhamento": "detalhe",
    "alterar": "alteracao",
    "alteracao": "alteracao",
    "disponibilizar": "disponibilizacao",
    "disponibilizacao": "disponibilizacao",
}

INTENCOES = {
    "cadastro": {
        "cadastro",
        "cadastrar",
        "incluir",
        "novo",
    },

    "alteracao": {
        "alteracao",
        "alterar",
        "editar",
    },

    "exclusao": {
        "exclusao",
        "excluir",
        "remover",
    },

    "ativacao": {
        "ativacao",
        "ativar",
    },

    "inativacao": {
        "inativacao",
        "inativar",
        "desativar",
    },

    "disponibilizacao": {
        "disponibilizacao",
        "disponibilizar",
    },
}

# ============================================================
# NORMALIZAÇÃO
# ============================================================

def remover_acentos(texto):
    texto = unicodedata.normalize("NFD", texto)
    return "".join(
        c for c in texto
        if unicodedata.category(c) != "Mn"
    )

def normalizar_texto(texto):
    if not texto:
        return ""

    texto = remover_acentos(texto.lower())
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()

def normalizar_termo(termo):
    termo = normalizar_texto(termo)

    if not termo:
        return ""

    if len(termo) > 4 and termo.endswith("s"):
        termo = termo[:-1]

    return EQUIVALENCIAS.get(termo, termo)

def extrair_termos(pergunta):
    return {
        normalizar_termo(p)
        for p in normalizar_texto(pergunta).split()
        if len(p) >= 3 and p not in STOPWORDS
    }

# ============================================================
# FAISS
# ============================================================

print("Carregando modelo de embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name=MODELO_EMBEDDING
)

print("Carregando índice FAISS...")

vectorstore = FAISS.load_local(
    PASTA_INDICE,
    embeddings,
    allow_dangerous_deserialization=True
)

todos_documentos = list(
    vectorstore.docstore._dict.values()
)

print(f"Documentos disponíveis: {len(todos_documentos)}")

# ============================================================
# IDENTIFICAÇÃO DO DOCUMENTO
# ============================================================

def chave_documento(documento: Document):
    return (
        documento.metadata.get("source", ""),
        documento.metadata.get("page", ""),
    )

def identificacao_documento(documento):
    fonte = os.path.splitext(
        os.path.basename(
            documento.metadata.get("source", "")
        )
    )[0]

    titulo = documento.metadata.get("title", "")

    return normalizar_texto(f"{fonte} {titulo}")

# ============================================================
# BUSCA TEXTUAL
# ============================================================

def busca_textual(pergunta, documentos, limite=K_TEXTUAL):
    termos = extrair_termos(pergunta)

    if not termos:
        return []

    resultados = []

    for documento in documentos:
        conteudo = normalizar_texto(documento.page_content)
        identificacao = identificacao_documento(documento)

        score = 0

        for termo in termos:
            if termo in identificacao:
                score += BOOST_TITULO

            ocorrencias = conteudo.count(termo)

            if ocorrencias:
                score += BOOST_CONTEUDO
                score += min(ocorrencias, 3) * BOOST_CONTEUDO

        # Regras específicas do domínio
        if "ente" in termos and "osc" in termos:
            if "ente osc" in conteudo:
                score += 3

        if "cadastro" in termos and "cadastro" in conteudo:
            score += 3

        if score > 0:
            resultados.append((documento, score))

    resultados.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return resultados[:limite]

# ============================================================
# BUSCA SEMÂNTICA
# ============================================================

def busca_semantica(pergunta, limite=K_SEMANTICA):
    resultados = vectorstore.similarity_search_with_score(
        pergunta,
        k=limite
    )

    return [
        (
            documento,
            max(0, limite - posicao)
        )
        for posicao, (documento, _) in enumerate(resultados)
    ]

# ============================================================
# BUSCA HÍBRIDA
# ============================================================

def buscar_documentos(pergunta):
    termos = extrair_termos(pergunta)

    print("\n" + "=" * 70)
    print("BUSCA HÍBRIDA")
    print("=" * 70)
    print(f"Pergunta: {pergunta}")
    print(f"Termos: {sorted(termos)}")

    semanticos = busca_semantica(pergunta)
    textuais = busca_textual(pergunta, todos_documentos)

    ranking = {}

    def adicionar(documento, semantico=0, textual=0):
        chave = chave_documento(documento)

        if chave not in ranking:
            ranking[chave] = {
                "documento": documento,
                "semantico": 0,
                "textual": 0,
                "boost": 0
            }

        ranking[chave]["semantico"] += semantico
        ranking[chave]["textual"] += textual

    for documento, score in semanticos:
        adicionar(
            documento,
            semantico=score * PESO_SEMANTICO
        )

    for documento, score in textuais:
        adicionar(
            documento,
            textual=score * PESO_TEXTUAL
        )

    # --------------------------------------------------------
    # BOOST POR TÍTULO
    # --------------------------------------------------------

    boosts = {
        "osc": 5,
        "chamamento": 8,
        "proposta": 8,
        "anexo": 8,
        "usuario": 5
    }

    for item in ranking.values():
        identificacao = identificacao_documento(
            item["documento"]
        )

        if "cadastro" in termos and "ente" in termos:
            if "cadastro" in identificacao and "ente" in identificacao:
                item["boost"] += 10

        for termo, valor in boosts.items():
            if termo in termos and termo in identificacao:
                item["boost"] += valor

        # --------------------------------------------------------
        # BOOST POR INTENÇÃO + OBJETO
        # --------------------------------------------------------

        for (intencao, objeto), valor in BOOST_INTENCAO_OBJETO.items():

            if intencao in termos and objeto in termos:

                if intencao in identificacao and objeto in identificacao:
                    item["boost"] += valor

        # --------------------------------------------------------
        # BOOST DE INTENÇÃO
        # --------------------------------------------------------

        intencoes = {
            "cadastro",
            "alteracao",
            "exclusao",
            "ativacao",
            "inativacao",
            "consulta",
            "detalhe",
            "disponibilizacao",
        }

        BOOST_INTENCAO_OBJETO = {
            ("cadastro", "usuario"): 20,
            ("cadastro", "ente"): 20,
            ("cadastro", "osc"): 20,

            ("alteracao", "chamamento"): 20,
            ("exclusao", "chamamento"): 20,
            ("ativacao", "chamamento"): 20,
            ("inativacao", "chamamento"): 20,
            ("disponibilizacao", "chamamento"): 20,

            ("alteracao", "proposta"): 20,
            ("exclusao", "usuario"): 20,
        }

        item["score"] = (
                item["semantico"]
                + item["textual"]
                + item["boost"]
        )

    # --------------------------------------------------------
    # RANKING
    # --------------------------------------------------------

    resultados = sorted(
        ranking.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    # ============================================================
    # FILTRO DE RELEVÂNCIA
    # ============================================================

    if resultados:
        maior_score = resultados[0]["score"]

        # Mantém apenas documentos com pelo menos 50% da pontuação do melhor resultado.
        resultados_relevantes = [
            item
            for item in resultados
            if item["score"] >= maior_score * 0.5
        ]
    else:
        resultados_relevantes = []

    # ============================================================
    # RANKING
    # ============================================================

    print("\nRanking:")

    for i, item in enumerate(resultados_relevantes[:K_FINAL], 1):
        documento = item["documento"]

        print(
            f"[{i}] "
            f"score={item['score']:.2f} | "
            f"sem={item['semantico']:.2f} | "
            f"text={item['textual']:.2f} | "
            f"boost={item['boost']:.2f} | "
            f"{documento.metadata.get('source')} "
            f"página={documento.metadata.get('page')}"
        )

    # ============================================================
    # DOCUMENTOS FINAIS
    # ============================================================

    documentos_finais = [
        item["documento"]
        for item in resultados_relevantes[:K_FINAL]
    ]

    print("\nDocumentos enviados ao LLM:")

    for i, documento in enumerate(documentos_finais, 1):
        print(
            f"[{i}] "
            f"{documento.metadata.get('source')} "
            f"página={documento.metadata.get('page')}"
        )

    return documentos_finais

# ============================================================
# LLM
# ============================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPENAI_API_KEY
)

# ============================================================
# PROMPT
# ============================================================

system_prompt = f"""
Você é um assistente especializado no sistema GTV.

Responda usando EXCLUSIVAMENTE as informações presentes
nos documentos fornecidos.

Regras:
- Não invente informações.
- Não use conhecimento externo.
- Se houver informação suficiente, responda normalmente.
- Se não houver informação suficiente, responda exatamente:
  {MARCADOR_NAO_SEI}
- Para procedimentos, mantenha a ordem das etapas.
- Pode combinar informações de documentos diferentes
  quando forem compatíveis.
- Não invente etapas.
- Não mencione FAISS, RAG, embeddings ou ranking.
- Seja direto e claro.
- Use listas numeradas para procedimentos.
- Não ultrapasse 7 frases, exceto quando etapas forem necessárias.

CONTEXTO:
{{context}}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(
    llm,
    prompt
)

# ============================================================
# RESPOSTA
# ============================================================

def answer_question(question):
    documentos = buscar_documentos(question)

    if not documentos:
        return MARCADOR_NAO_SEI

    resposta = question_answer_chain.invoke({
        "input": question,
        "context": documentos
    })

    return resposta.strip()

# ============================================================
# ATENDIMENTO HUMANO
# ============================================================

def precisa_de_atendente_humano(resposta):
    return resposta.strip().upper() == MARCADOR_NAO_SEI