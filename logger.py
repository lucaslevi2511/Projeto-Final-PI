import logging
import sys

# Configuração global do logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')  # Opcional: salvar em arquivo
    ]
)

# Logger global
logger = logging.getLogger(__name__)

# Função para desativar/ativar logs globalmente
def disable_logging():
    logging.disable(logging.CRITICAL)

def enable_logging():
    logging.disable(logging.NOTSET)