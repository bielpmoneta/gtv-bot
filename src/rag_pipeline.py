"""
Pipeline de RAG (Retrieval-Augmented Generation).

IMPORTANTE — leia antes de mexer:
Tudo que está FORA da função answer_question() roda UMA VEZ SÓ,
no momento em que esse módulo é importado (ex: quando o FastAPI sobe).
Isso inclui carregar o índice e montar a chain do LangChain.

A função answer_question() é a ÚNICA coisa que roda a cada pergunta —
e ela só faz busca no índice já carregado + uma chamada ao LLM.
Isso é o que evita gastar tokens/reprocessamento à toa.

Pré-requisito: rodar "python -m src.build_index" pelo menos uma vez
antes de usar esse módulo (precisa existir a pasta data/faiss_index).
"""

from config.config import OPENAI_API_KEY
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

PASTA_INDICE = "data/faiss_index"
NOME_MODELO_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Texto sentinela que o LLM retorna quando não encontra a resposta
# no contexto. O webhook usa isso para decidir se aciona o atendente humano.
MARCADOR_NAO_SEI = "NAO_SEI"

# ---------------------------------------------------------------
# Tudo abaixo roda UMA VEZ, na importação do módulo (não a cada pergunta)
# ---------------------------------------------------------------

# Mesmo modelo de embeddings usado na indexação — isso é OBRIGATÓRIO,
# senão os vetores gerados na consulta não são comparáveis com os do índice.
embeddings = HuggingFaceEmbeddings(model_name=NOME_MODELO_EMBEDDING)

# Carrega o índice já pronto do disco. allow_dangerous_deserialization=True
# é necessário porque o FAISS usa pickle internamente — é seguro aqui
# porque SOMOS NÓS que geramos esse arquivo (não veio de fonte externa).
vectorstore = FAISS.load_local(
    PASTA_INDICE,
    embeddings,
    allow_dangerous_deserialization=True,
)

# k=3 -> busca só os 3 pedaços mais relevantes para cada pergunta.
# É esse número que controla quanto contexto (= quantos tokens) vai
# pro LLM a cada chamada. Comece com 3; se as respostas vierem
# incompletas, tente subir para 4 ou 5.
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

system_prompt = (
    "Você é um assistente do sistema GTV, especializado em tirar dúvidas "
    "sobre documentos. Responda a pergunta do usuário usando SOMENTE as "
    "informações contidas no contexto abaixo. Não use conhecimento próprio, "
    "não invente nem complete informações que não estejam explicitamente "
    f"no contexto. Se a resposta não estiver clara no contexto, responda "
    f"exatamente '{MARCADOR_NAO_SEI}' e nada mais — sem explicações extras. "
    "Quando conseguir responder, seja direto e conciso, no máximo 3 frases."
    "\n\n"
    "Contexto:\n{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ---------------------------------------------------------------
# Isso roda a cada pergunta
# ---------------------------------------------------------------


def answer_question(question: str) -> str:
    """
    Busca os trechos mais relevantes no índice já carregado e gera
    a resposta com o LLM. Retorna o marcador NAO_SEI (sem alterações)
    quando o modelo não encontrou a resposta no contexto — é esse
    retorno que o webhook usa para decidir se aciona um atendente humano.
    """
    resultado = rag_chain.invoke({"input": question})
    return resultado["answer"].strip()


def precisa_de_atendente_humano(resposta: str) -> bool:
    """Função utilitária: centraliza a checagem do marcador NAO_SEI."""
    return resposta.strip().upper() == MARCADOR_NAO_SEI