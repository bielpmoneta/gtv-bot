"""
Pipeline de RAG híbrido do GTV.

Fluxo:

Pergunta
    ↓
Normalização
    ↓
Busca semântica FAISS
    ↓
Busca textual
    ↓
Boost por título / nome do documento
    ↓
Ranking híbrido
    ↓
Top documentos
    ↓
LLM
    ↓
Resposta / NAO_SEI

IMPORTANTE:

O índice FAISS precisa ser gerado previamente através de:

    python -m src.build_index

A indexação dos PDFs não acontece neste arquivo.
"""


import os
import re
import unicodedata
from collections import defaultdict

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
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# Marcador utilizado pelo webhook para decidir se deve
# encaminhar para atendimento humano.
MARCADOR_NAO_SEI = "NAO_SEI"


# ------------------------------------------------------------
# BUSCA
# ------------------------------------------------------------

# Quantos documentos o FAISS recupera inicialmente.
K_BUSCA_SEMANTICA = 15

# Quantos documentos a busca textual recupera.
K_BUSCA_TEXTUAL = 15

# Quantos documentos serão enviados ao LLM.
K_DOCUMENTOS_FINAIS = 5


# ------------------------------------------------------------
# PESOS
# ------------------------------------------------------------

# Peso da posição na busca semântica.
PESO_SEMANTICO = 1.0

# Peso da posição na busca textual.
PESO_TEXTUAL = 1.5

# Pontos extras quando palavras importantes aparecem
# no nome/título do documento.
BOOST_TITULO = 5.0

# Pontos extras quando palavras importantes aparecem
# no início do conteúdo do chunk.
BOOST_CONTEUDO = 0.5


# ============================================================
# STOPWORDS
# ============================================================

# Palavras muito genéricas não devem influenciar muito
# o ranking textual.

STOPWORDS = {
    "a",
    "à",
    "ao",
    "aos",
    "as",
    "com",
    "como",
    "da",
    "das",
    "de",
    "do",
    "dos",
    "e",
    "em",
    "essa",
    "esse",
    "esta",
    "este",
    "eu",
    "fazer",
    "faço",
    "foi",
    "for",
    "há",
    "isso",
    "isto",
    "na",
    "nas",
    "no",
    "nos",
    "o",
    "os",
    "ou",
    "para",
    "por",
    "que",
    "se",
    "sem",
    "ser",
    "um",
    "uma",
    "umas",
    "uns",
    "uso",
}


# ============================================================
# NORMALIZAÇÃO
# ============================================================

def remover_acentos(texto: str) -> str:
    """
    Remove acentos.

    Exemplo:

        organização
        organização

    vira:

        organizacao
    """

    texto = unicodedata.normalize(
        "NFD",
        texto,
    )

    return "".join(
        caractere
        for caractere in texto
        if unicodedata.category(caractere) != "Mn"
    )


def normalizar_texto(texto: str) -> str:
    """
    Normaliza texto para comparação.

    Exemplos:

        "Ente/OSC"
        "ENTE OSC"
        "ente-osc"

    ficam mais próximos para a busca textual.
    """

    if not texto:
        return ""

    texto = texto.lower()

    texto = remover_acentos(texto)

    # Mantém letras, números e espaços.
    texto = re.sub(
        r"[^a-z0-9\s]",
        " ",
        texto,
    )

    texto = re.sub(
        r"\s+",
        " ",
        texto,
    )

    return texto.strip()


# ============================================================
# NORMALIZAÇÃO DE TERMOS DO GTV
# ============================================================

def normalizar_termo(termo: str) -> str:
    """
    Normaliza pequenas variações comuns das palavras.

    Isso ajuda a aproximar perguntas como:

        cadastrar
        cadastro
        cadastramento

        ente
        entes

        usuario
        usuarios
    """

    termo = normalizar_texto(termo)

    if not termo:
        return ""

    # Plurais simples
    if len(termo) > 4 and termo.endswith("s"):
        termo = termo[:-1]

    # Variações muito comuns no domínio do sistema.
    equivalencias = {
        "cadastrar": "cadastro",
        "cadastramento": "cadastro",
        "cadastrada": "cadastro",
        "cadastrado": "cadastro",
        "cadastros": "cadastro",

        "entes": "ente",

        "osc": "osc",

        "usuarios": "usuario",
        "usuaria": "usuario",
        "usuarias": "usuario",

        "convenentes": "convenente",

        "chamamentos": "chamamento",

        "propostas": "proposta",

        "anexos": "anexo",
    }

    return equivalencias.get(
        termo,
        termo,
    )


def extrair_termos_importantes(pergunta: str) -> set[str]:
    """
    Extrai palavras relevantes da pergunta.

    Remove stopwords e normaliza termos.
    """

    texto = normalizar_texto(pergunta)

    palavras = texto.split()

    termos = set()

    for palavra in palavras:

        if len(palavra) < 3:
            continue

        if palavra in STOPWORDS:
            continue

        termo = normalizar_termo(palavra)

        if termo:
            termos.add(termo)

    return termos


# ============================================================
# EMBEDDINGS
# ============================================================

print("=" * 70)
print("CARREGANDO MODELO DE EMBEDDINGS")
print("=" * 70)

embeddings = HuggingFaceEmbeddings(
    model_name=NOME_MODELO_EMBEDDING
)


# ============================================================
# FAISS
# ============================================================

print("\nCarregando índice FAISS...")

vectorstore = FAISS.load_local(
    PASTA_INDICE,
    embeddings,
    allow_dangerous_deserialization=True,
)


# ============================================================
# DOCUMENTOS DO ÍNDICE
# ============================================================

# Recuperamos os documentos diretamente do docstore.
#
# Isso permite fazer uma busca textual local sem precisar
# chamar o FAISS para essa etapa.

todos_documentos = list(
    vectorstore.docstore._dict.values()
)

print(
    f"Documentos disponíveis para busca textual: "
    f"{len(todos_documentos)}"
)


# ============================================================
# CHAVE ÚNICA DO DOCUMENTO
# ============================================================

def chave_documento(documento: Document) -> tuple:
    """
    Cria uma chave estável para evitar documentos duplicados.
    """

    fonte = documento.metadata.get(
        "source",
        "",
    )

    pagina = documento.metadata.get(
        "page",
        "",
    )

    conteudo = documento.page_content[:150]

    return (
        fonte,
        pagina,
        conteudo,
    )


# ============================================================
# BUSCA TEXTUAL
# ============================================================

def busca_textual(
    pergunta: str,
    documentos: list[Document],
    limite: int = K_BUSCA_TEXTUAL,
) -> list[tuple[Document, float]]:
    """
    Busca documentos utilizando correspondência textual.

    A pontuação considera:

    - presença dos termos importantes;
    - frequência dos termos;
    - ocorrência no título/nome do PDF;
    - ocorrência no conteúdo.
    """

    termos = extrair_termos_importantes(
        pergunta
    )

    resultados = []

    if not termos:
        return []

    for documento in documentos:

        conteudo = normalizar_texto(
            documento.page_content
        )

        fonte = normalizar_texto(
            os.path.splitext(
                os.path.basename(
                    documento.metadata.get(
                        "source",
                        "",
                    )
                )
            )[0]
        )

        titulo = normalizar_texto(
            documento.metadata.get(
                "title",
                "",
            )
        )

        identificacao = (
            f"{fonte} {titulo}"
        )

        score = 0.0

        # ----------------------------------------------------
        # TERMOS
        # ----------------------------------------------------

        for termo in termos:

            # -----------------------------------------------
            # TÍTULO / NOME DO DOCUMENTO
            # -----------------------------------------------

            if termo in identificacao:
                score += BOOST_TITULO

            # -----------------------------------------------
            # CONTEÚDO
            # -----------------------------------------------

            ocorrencias = conteudo.count(
                termo
            )

            if ocorrencias > 0:

                # Primeira ocorrência vale mais.
                score += BOOST_CONTEUDO

                # Repetições também ajudam, mas limitadas.
                score += min(
                    ocorrencias,
                    5,
                ) * 0.5

        # ----------------------------------------------------
        # FRASES IMPORTANTES
        # ----------------------------------------------------

        pergunta_normalizada = normalizar_texto(
            pergunta
        )

        # Busca pela combinação "ente" + "osc".
        if (
            "ente" in termos
            and "osc" in termos
            and "ente osc" in conteudo
        ):
            score += 3.0

        # Busca por intenção de cadastro.
        if (
            "cadastro" in termos
            and "cadastro" in conteudo
        ):
            score += 3.0

        # ----------------------------------------------------
        # ADICIONA RESULTADO
        # ----------------------------------------------------

        if score > 0:

            resultados.append(
                (
                    documento,
                    score,
                )
            )

    # Maior score primeiro.
    resultados.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return resultados[:limite]


# ============================================================
# BUSCA SEMÂNTICA
# ============================================================

def busca_semantica(
    pergunta: str,
    limite: int = K_BUSCA_SEMANTICA,
) -> list[tuple[Document, float]]:
    """
    Busca semântica utilizando FAISS.

    Retorna documento + score normalizado por posição.

    O FAISS retorna distância, portanto não usamos a distância
    diretamente como pontuação final. Usamos a posição do
    documento no ranking.
    """

    resultados = (
        vectorstore.similarity_search_with_score(
            pergunta,
            k=limite,
        )
    )

    resultados_formatados = []

    for posicao, (documento, distancia) in enumerate(
        resultados
    ):

        # Ranking decrescente.
        #
        # Primeiro resultado:
        #   15 pontos
        #
        # Segundo:
        #   14 pontos
        #
        # etc.

        score = max(
            0.0,
            float(limite - posicao),
        )

        resultados_formatados.append(
            (
                documento,
                score,
            )
        )

    return resultados_formatados


# ============================================================
# BUSCA HÍBRIDA
# ============================================================

def buscar_documentos(
    pergunta: str,
) -> list[Document]:
    """
    Executa busca híbrida:

        FAISS + busca textual + título

    e retorna os documentos mais relevantes.
    """

    # ========================================================
    # TERMOS
    # ========================================================

    termos = extrair_termos_importantes(
        pergunta
    )

    print("\n")
    print("=" * 70)
    print("BUSCA HÍBRIDA")
    print("=" * 70)

    print(
        f"Pergunta: {pergunta}"
    )

    print(
        f"Termos importantes: {sorted(termos)}"
    )

    # ========================================================
    # SEMÂNTICA
    # ========================================================

    resultados_semanticos = busca_semantica(
        pergunta
    )

    # ========================================================
    # TEXTUAL
    # ========================================================

    resultados_textuais = busca_textual(
        pergunta,
        todos_documentos,
    )

    # ========================================================
    # PONTUAÇÕES
    # ========================================================

    pontuacoes = {}

    # --------------------------------------------------------
    # SEMÂNTICA
    # --------------------------------------------------------

    for documento, score in resultados_semanticos:

        chave = chave_documento(
            documento
        )

        if chave not in pontuacoes:

            pontuacoes[chave] = {
                "documento": documento,
                "score_semantico": 0.0,
                "score_textual": 0.0,
                "boost": 0.0,
            }

        pontuacoes[chave][
            "score_semantico"
        ] += score * PESO_SEMANTICO

    # --------------------------------------------------------
    # TEXTUAL
    # --------------------------------------------------------

    for documento, score in resultados_textuais:

        chave = chave_documento(
            documento
        )

        if chave not in pontuacoes:

            pontuacoes[chave] = {
                "documento": documento,
                "score_semantico": 0.0,
                "score_textual": 0.0,
                "boost": 0.0,
            }

        pontuacoes[chave][
            "score_textual"
        ] += score * PESO_TEXTUAL

    # ========================================================
    # BOOST ESPECIAL
    # ========================================================

    for item in pontuacoes.values():

        documento = item["documento"]

        fonte = normalizar_texto(
            os.path.splitext(
                os.path.basename(
                    documento.metadata.get(
                        "source",
                        "",
                    )
                )
            )[0]
        )

        titulo = normalizar_texto(
            documento.metadata.get(
                "title",
                "",
            )
        )

        identificacao = (
            f"{fonte} {titulo}"
        )

        # ----------------------------------------------------
        # CADASTRO DE ENTE / OSC
        # ----------------------------------------------------

        if (
            "cadastro" in termos
            and "ente" in termos
        ):

            if (
                "cadastro" in identificacao
                and "ente" in identificacao
            ):
                item["boost"] += 10.0

        # ----------------------------------------------------
        # OSC
        # ----------------------------------------------------

        if (
            "osc" in termos
            and "osc" in identificacao
        ):
            item["boost"] += 5.0

        # ----------------------------------------------------
        # CHAMAMENTO
        # ----------------------------------------------------

        if (
            "chamamento" in termos
            and "chamamento" in identificacao
        ):
            item["boost"] += 8.0

        # ----------------------------------------------------
        # PROPOSTA
        # ----------------------------------------------------

        if (
            "proposta" in termos
            and "proposta" in identificacao
        ):
            item["boost"] += 8.0

        # ----------------------------------------------------
        # ANEXO
        # ----------------------------------------------------

        if (
            "anexo" in termos
            and "anexo" in identificacao
        ):
            item["boost"] += 8.0

        # ----------------------------------------------------
        # USUÁRIO
        # ----------------------------------------------------

        if (
            "usuario" in termos
            and "usuario" in identificacao
        ):
            item["boost"] += 5.0

    # ========================================================
    # SCORE FINAL
    # ========================================================

    resultados_finais = []

    for item in pontuacoes.values():

        score_final = (
            item["score_semantico"]
            + item["score_textual"]
            + item["boost"]
        )

        item["score_final"] = score_final

        resultados_finais.append(
            item
        )

    # Maior score primeiro.
    resultados_finais.sort(
        key=lambda item: item["score_final"],
        reverse=True,
    )

    # ========================================================
    # DEBUG
    # ========================================================

    print("\n")
    print("=" * 70)
    print("RANKING DOS DOCUMENTOS")
    print("=" * 70)

    for posicao, item in enumerate(
        resultados_finais[:K_DOCUMENTOS_FINAIS],
        start=1,
    ):

        documento = item["documento"]

        print(
            f"\n[{posicao}] "
            f"score={item['score_final']:.2f}"
        )

        print(
            f"  semântico = "
            f"{item['score_semantico']:.2f}"
        )

        print(
            f"  textual   = "
            f"{item['score_textual']:.2f}"
        )

        print(
            f"  boost     = "
            f"{item['boost']:.2f}"
        )

        print(
            f"  fonte     = "
            f"{documento.metadata.get('source')}"
        )

        print(
            f"  página    = "
            f"{documento.metadata.get('page')}"
        )

    # ========================================================
    # DOCUMENTOS FINAIS
    # ========================================================

    documentos_finais = [
        item["documento"]
        for item in resultados_finais[
            :K_DOCUMENTOS_FINAIS
        ]
    ]

    print("\n")
    print("=" * 70)
    print("DOCUMENTOS ENVIADOS AO LLM")
    print("=" * 70)

    for i, documento in enumerate(
        documentos_finais,
        start=1,
    ):

        print(
            f"[{i}] "
            f"{documento.metadata.get('source')} "
            f"página={documento.metadata.get('page')}"
        )

    print("=" * 70)

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

1. Use SOMENTE as informações presentes no contexto.

2. Não invente informações.

3. Não utilize conhecimento externo.

4. Se o contexto possuir informação suficiente para responder,
responda normalmente.

5. Se não houver informação suficiente, responda exatamente:

{MARCADOR_NAO_SEI}

6. Não retorne {MARCADOR_NAO_SEI} quando a resposta puder
ser obtida com segurança a partir do contexto.

7. Para perguntas sobre procedimentos, apresente as etapas
na ordem em que aparecem nos documentos.

8. Se existirem informações complementares em documentos
diferentes, você pode combiná-las, desde que elas sejam
compatíveis entre si.

9. Não invente etapas que não estejam descritas.

10. Não mencione FAISS, RAG, embeddings, busca semântica,
ranking ou contexto interno.

11. Seja direto e claro.

12. Prefira listas numeradas quando estiver explicando
procedimentos.

13. Não ultrapasse 5 frases, exceto quando uma lista de
etapas for necessária.

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
    Executa a busca híbrida e gera a resposta.
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

    resposta = resultado.strip()

    return resposta


# ============================================================
# ATENDIMENTO HUMANO
# ============================================================

def precisa_de_atendente_humano(
    resposta: str,
) -> bool:
    """
    Verifica se o RAG não conseguiu responder.
    """

    return (
        resposta.strip().upper()
        == MARCADOR_NAO_SEI
    )