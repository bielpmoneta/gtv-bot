"""
Rota que recebe os eventos da Evolution API, gera a resposta usando
o pipeline RAG, e manda a resposta de volta para o WhatsApp do usuário.
"""

from fastapi import APIRouter, Request
from src.rag_pipeline import answer_question, precisa_de_atendente_humano
from src.client import enviar_mensagem

router = APIRouter()

# Controle simples de quem está em standby aguardando atendente humano.
# ATENÇÃO: isso é só um placeholder por enquanto — é um dicionário em
# memória, então some se o servidor reiniciar, e não tem timeout ainda.
# O próximo passo do roteiro (handoff humano de verdade) troca isso
# por uma tabela no banco de dados.
usuarios_em_standby: dict[str, bool] = {}

MENSAGEM_TRANSFERENCIA = (
    "Não encontrei essa informação nos documentos do GTV. "
    "Vou te transferir para um atendente humano, só um momento."
)


@router.post("/webhook")
async def receber_webhook(request: Request):
    payload = await request.json()
    evento = payload.get("event")

    if evento != "messages.upsert":
        # Outros eventos (status de conexão, confirmação de leitura, etc)
        # não interessam pro chatbot por enquanto.
        return {"status": "ignorado"}

    dados = payload.get("data", {})

    numero_remetente = dados.get("key", {}).get("remoteJid")
    texto_mensagem = dados.get("message", {}).get("conversation")

    # Mensagens sem texto (áudio, imagem, figurinha) ou sem remetente
    # identificável não têm o que processar por enquanto.
    if not numero_remetente or not texto_mensagem:
        return {"status": "ignorado"}

    # Ignora mensagens enviadas pelo próprio bot (eco), evitando loop.
    if dados.get("key", {}).get("fromMe"):
        return {"status": "ignorado"}

    # Se o usuário já está em standby aguardando humano, o bot não
    # responde mais nessa conversa — só um humano assume a partir daqui.
    if usuarios_em_standby.get(numero_remetente):
        return {"status": "em_standby"}

    resposta = answer_question(texto_mensagem)

    if precisa_de_atendente_humano(resposta):
        usuarios_em_standby[numero_remetente] = True
        enviar_mensagem(numero_remetente, MENSAGEM_TRANSFERENCIA)
        return {"status": "transferido_para_humano"}

    enviar_mensagem(numero_remetente, resposta)
    return {"status": "respondido"}