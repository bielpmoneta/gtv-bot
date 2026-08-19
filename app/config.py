# variáveis de ambiente (API key, URL da evolution, etc)

"""
Centraliza a leitura das variáveis de ambiente.
Assim, o resto do código nunca acessa os.getenv diretamente —
só importa as constantes daqui. Isso facilita muito quando
o projeto crescer (e evita repetir os.getenv espalhado pelo código).
"""

import os
from dotenv import load_dotenv

# Carrega o conteúdo do arquivo .env para as variáveis de ambiente do processo Python. Precisa ser chamado antes de os.getenv().
load_dotenv()

# URL onde a Evolution API está rodando (a mesma que você usa no Manager)
EVOLUTION_API_URL = os.getenv("SERVER_URL", "")

# API Key da Evolution API (a mesma configurada no .env dela)
EVOLUTION_API_KEY = os.getenv("EVOLUTION_API_KEY", "")

# Nome da instância que você criou no Manager (ex: "gtv-bot")
EVOLUTION_INSTANCE_NAME = os.getenv("EVOLUTION_INSTANCE_NAME", "")

# Validação simples: se alguma variável essencial estiver faltando, o app já avisa no console em vez de falhar silenciosamente depois.
if not EVOLUTION_API_KEY:
    print("[AVISO] EVOLUTION_API_KEY não está definida no .env")

if not EVOLUTION_INSTANCE_NAME:
    print("[AVISO] EVOLUTION_INSTANCE_NAME não está definida no .env")









