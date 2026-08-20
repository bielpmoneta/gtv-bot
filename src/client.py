"""
Cliente simples para chamar a Evolution API e enviar mensagens de
texto de volta para o WhatsApp do usuário.
"""

import requests
from config.config import EVOLUTION_API_URL, EVOLUTION_API_KEY, EVOLUTION_INSTANCE_NAME


def enviar_mensagem(numero: str, texto: str) -> None:
    """
    Envia uma mensagem de texto para o número informado via Evolution API.

    numero: no formato que vem no remoteJid, ex "5511999999999@s.whatsapp.net"
            ou só os dígitos "5511999999999" — a Evolution API aceita os dois,
            mas por clareza extraímos só os dígitos antes de enviar.
    """
    numero_limpo = numero.split("@")[0]

    url = f"{EVOLUTION_API_URL}/message/sendText/{EVOLUTION_INSTANCE_NAME}"
    headers = {
        "Content-Type": "application/json",
        "apikey": EVOLUTION_API_KEY,
    }
    payload = {
        "number": numero_limpo,
        "text": texto,
    }

    resposta = requests.post(url, json=payload, headers=headers, timeout=15)

    if resposta.status_code not in (200, 201):
        # Não derruba a aplicação por causa disso — só loga o erro.
        # Em produção, isso seria um bom lugar para alertar você de alguma forma.
        print(f"[ERRO] Falha ao enviar mensagem para {numero_limpo}: "
              f"{resposta.status_code} - {resposta.text}")