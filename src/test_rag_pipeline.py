"""
Testes para o pipeline de RAG do GTV.

IMPORTANTE:
Este arquivo assume que o módulo do pipeline se chama `rag_pipeline.py`
e fica na raiz do projeto (ajuste o import abaixo para o nome/caminho real,
ex.: `from app.rag_pipeline import ...`).

Como o módulo original instancia vectorstore, embeddings e llm no nível
de módulo (fora de funções), fazemos o mock ANTES do import, usando
sys.modules e monkeypatch, para não precisar do índice FAISS real,
do modelo de embeddings baixado, nem de uma chave OpenAI válida.

Rodar:
    pip install pytest --break-system-packages
    pytest test_rag_pipeline.py -v
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

# Garante que a pasta onde este arquivo de teste está (e onde
# rag_pipeline.py deve estar) esteja no sys.path, independente
# de onde o comando `pytest` for executado no terminal.
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ============================================================
# FIXTURE: mocka dependências pesadas ANTES do import do módulo
# ============================================================

@pytest.fixture
def rag_module():
    """
    Importa o módulo do pipeline com FAISS, embeddings e LLM mockados,
    evitando I/O real em disco/rede durante os testes.
    """

    # Mocka a classe HuggingFaceEmbeddings para não baixar/carregar modelo.
    mock_embeddings_cls = MagicMock()

    # Mocka FAISS.load_local para não precisar do índice real em disco.
    mock_vectorstore = MagicMock()
    mock_vectorstore.docstore._dict = {}
    mock_vectorstore.similarity_search.return_value = []

    mock_faiss_cls = MagicMock()
    mock_faiss_cls.load_local.return_value = mock_vectorstore

    # Mocka ChatOpenAI para não precisar de API key/rede.
    mock_llm_cls = MagicMock()

    with patch(
        "langchain_huggingface.HuggingFaceEmbeddings",
        mock_embeddings_cls,
    ), patch(
        "langchain_community.vectorstores.FAISS",
        mock_faiss_cls,
    ), patch(
        "langchain_openai.ChatOpenAI",
        mock_llm_cls,
    ), patch(
        "config.config.OPENAI_API_KEY",
        "fake-key",
        create=True,
    ):
        # Remove do cache para forçar reimport com os mocks ativos.
        sys.modules.pop("rag_pipeline", None)
        import rag_pipeline  # ajuste este nome/caminho ao seu projeto

        yield rag_pipeline, mock_vectorstore


# ============================================================
# TESTES: normalizar_texto (função pura)
# ============================================================

class TestNormalizarTexto:

    def test_minusculas(self, rag_module):
        modulo, _ = rag_module
        assert modulo.normalizar_texto("TESTE") == "teste"

    def test_remove_pontuacao(self, rag_module):
        modulo, _ = rag_module
        resultado = modulo.normalizar_texto("Olá, mundo!")
        assert resultado == "olá mundo"

    def test_colapsa_espacos(self, rag_module):
        modulo, _ = rag_module
        resultado = modulo.normalizar_texto("a    b\t\tc")
        assert resultado == "a b c"

    def test_string_vazia(self, rag_module):
        modulo, _ = rag_module
        assert modulo.normalizar_texto("") == ""

    def test_mantem_acentos(self, rag_module):
        modulo, _ = rag_module
        # acentos não são removidos, só pontuação
        assert "não" in modulo.normalizar_texto("Não sei")


# ============================================================
# TESTES: busca_textual (função pura, recebe lista de Document)
# ============================================================

class TestBuscaTextual:

    def _docs(self):
        return [
            Document(page_content="Como cadastrar um veículo no GTV"),
            Document(page_content="Procedimento de manutenção preventiva"),
            Document(page_content="Cadastro de motorista e veículo"),
            Document(page_content="Receita de bolo de cenoura"),
        ]

    def test_encontra_documentos_relevantes(self, rag_module):
        modulo, _ = rag_module
        resultados = modulo.busca_textual(
            "como cadastrar veículo",
            self._docs(),
        )
        assert len(resultados) > 0
        assert "veículo" in modulo.normalizar_texto(
            resultados[0].page_content
        )

    def test_ordena_por_pontuacao(self, rag_module):
        modulo, _ = rag_module
        resultados = modulo.busca_textual(
            "cadastro veículo",
            self._docs(),
        )
        # o doc com "Cadastro de motorista e veículo" bate as duas palavras
        assert "Cadastro de motorista e veículo" in [
            d.page_content for d in resultados[:2]
        ]

    def test_sem_correspondencia_retorna_vazio(self, rag_module):
        modulo, _ = rag_module
        resultados = modulo.busca_textual(
            "assunto totalmente ausente xyz123",
            self._docs(),
        )
        assert resultados == []

    def test_respeita_limite(self, rag_module):
        modulo, _ = rag_module
        docs = [
            Document(page_content=f"veículo número {i}") for i in range(20)
        ]
        resultados = modulo.busca_textual("veículo", docs, limite=3)
        assert len(resultados) == 3

    def test_ignora_palavras_curtas(self, rag_module):
        modulo, _ = rag_module
        # palavras com menos de 3 letras (ex: "um", "de") são ignoradas
        resultados = modulo.busca_textual(
            "um de e a",
            self._docs(),
        )
        assert resultados == []


# ============================================================
# TESTES: precisa_de_atendente_humano (função pura)
# ============================================================

class TestPrecisaDeAtendenteHumano:

    def test_marcador_exato(self, rag_module):
        modulo, _ = rag_module
        assert modulo.precisa_de_atendente_humano("NAO_SEI") is True

    def test_marcador_com_espacos(self, rag_module):
        modulo, _ = rag_module
        assert modulo.precisa_de_atendente_humano("  NAO_SEI  ") is True

    def test_marcador_minusculo(self, rag_module):
        modulo, _ = rag_module
        # a função faz .upper(), então deve reconhecer mesmo em minúsculo
        assert modulo.precisa_de_atendente_humano("nao_sei") is True

    def test_resposta_normal_nao_aciona(self, rag_module):
        modulo, _ = rag_module
        resposta = "Para cadastrar um veículo, acesse o menu Cadastros."
        assert modulo.precisa_de_atendente_humano(resposta) is False


# ============================================================
# TESTES: buscar_documentos (integração com FAISS mockado)
# ============================================================

class TestBuscarDocumentos:

    def test_combina_semantico_e_textual_sem_duplicar(self, rag_module):
        modulo, mock_vectorstore = rag_module

        doc_comum = Document(
            page_content="Como cadastrar veículo no sistema GTV",
            metadata={"source": "manual.pdf", "page": 1},
        )
        doc_so_semantico = Document(
            page_content="Outro conteúdo qualquer",
            metadata={"source": "manual.pdf", "page": 2},
        )

        mock_vectorstore.similarity_search.return_value = [
            doc_comum,
            doc_so_semantico,
        ]
        # busca_textual usa todos_documentos, que é populado no import;
        # para simplificar, testamos aqui via monkeypatch direto:
        modulo.todos_documentos = [doc_comum]

        resultados = modulo.buscar_documentos("cadastrar veículo")

        # doc_comum não deve aparecer duplicado mesmo vindo dos dois caminhos
        chaves = [d.page_content for d in resultados]
        assert chaves.count(doc_comum.page_content) == 1

    def test_respeita_limite_final(self, rag_module):
        modulo, mock_vectorstore = rag_module

        docs = [
            Document(
                page_content=f"conteúdo {i}",
                metadata={"source": "a.pdf", "page": i},
            )
            for i in range(20)
        ]
        mock_vectorstore.similarity_search.return_value = docs
        modulo.todos_documentos = []

        resultados = modulo.buscar_documentos("qualquer pergunta")
        assert len(resultados) <= modulo.K_DOCUMENTOS_FINAIS


# ============================================================
# TESTES: answer_question (integração com chain mockado)
# ============================================================

class TestAnswerQuestion:

    def test_sem_documentos_retorna_nao_sei(self, rag_module):
        modulo, mock_vectorstore = rag_module
        mock_vectorstore.similarity_search.return_value = []
        modulo.todos_documentos = []

        resultado = modulo.answer_question("pergunta qualquer")
        assert resultado == modulo.MARCADOR_NAO_SEI

    def test_com_documentos_chama_chain_e_retorna_resposta(self, rag_module):
        modulo, mock_vectorstore = rag_module

        doc = Document(
            page_content="Para cadastrar um veículo, acesse Cadastros > Veículos.",
            metadata={"source": "manual.pdf", "page": 1},
        )
        mock_vectorstore.similarity_search.return_value = [doc]
        modulo.todos_documentos = [doc]

        # mocka a chain para não chamar a API da OpenAI de verdade
        mock_resultado = MagicMock()
        mock_resultado.strip.return_value = "Acesse Cadastros > Veículos."
        modulo.question_answer_chain = MagicMock()
        modulo.question_answer_chain.invoke.return_value = "Acesse Cadastros > Veículos."

        resultado = modulo.answer_question("como cadastrar veículo?")

        modulo.question_answer_chain.invoke.assert_called_once()
        assert resultado == "Acesse Cadastros > Veículos."