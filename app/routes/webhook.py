"""
Rota responsável por receber os eventos que a Evolution API envia
via webhook (mensagens recebidas, atualizações de status, etc).

Por enquanto, essa rota só IMPRIME o payload no console.
O objetivo dessa etapa é você enxergar o formato real que a Evolution
API manda, antes de escrever qualquer lógica de negócio em cima disso.
"""

from fastapi import APIRouter, Request
import json

router = APIRouter()

@router.post("/webhook")
async def receber_webhook(request: Request):
    # Pega o corpo da requisição como JSON (dict Python)
    payload = await request.json()

    # Imprime formatado (indent=2) pra ficar legível no terminal
    print("\n===== NOVO EVENTO RECEBIDO =====")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("=================================\n")

    # A Evolution API sempre manda um campo "event" identificando
    # o tipo de evento. O que mais nos interessa por enquanto é
    # "messages.upsert" (nova mensagem recebida).
    evento = payload.get("event")

    if evento == "messages.upsert":
        # A estrutura real da mensagem vem dentro de "data".
        # Isso pode variar um pouco dependendo da versão da Evolution API,
        # por isso usamos .get() em vez de acesso direto com [] —
        # assim, se algum campo não existir, não quebra a aplicação.
        dados = payload.get("data", {})

        numero_remetente = dados.get("key", {}).get("remoteJid")
        texto_mensagem = dados.get("message", {}).get("conversation")

        print(f"Número: {numero_remetente}")
        print(f"Mensagem: {texto_mensagem}")

    # A Evolution API espera algum retorno HTTP 200 para saber que
    # recebemos o evento com sucesso. O conteúdo do retorno não importa
    # muito aqui, mas é boa prática sempre devolver algo.
    return {"status": "recebido"}