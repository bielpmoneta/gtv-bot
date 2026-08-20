"""
Ponto de entrada da aplicação. É esse arquivo que o uvicorn
carrega para subir o servidor (uvicorn app.main:app --reload).
"""

from fastapi import FastAPI
from src import webhook

app = FastAPI(title="GTV Bot")

# Registra as rotas definidas em webhook.py dentro da aplicação principal.
# Sem essa linha, o FastAPI nem saberia que a rota /webhook existe.
app.include_router(webhook.router)

@app.get("/")
async def health_check():
    """
    Rota simples só pra confirmar que o servidor está de pé.
    Útil para testar rapidamente no navegador: http://localhost:8000
    """
    return {"status": "online", "servico": "GTV Bot"}