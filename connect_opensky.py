# connect_opensky.py

import os
import logging
import trino
from trino.auth import OAuth2Authentication, ConsoleRedirectHandler
from dotenv import load_dotenv
load_dotenv()  # carrega o .env para as variáveis de ambiente

logger = logging.getLogger(__name__)

# Variáveis globais para singleton
_conn = None
_cur  = None

def create_connection():
    """
    Retorna um singleton (conn, cursor) para o Trino/OpenSky usando OAuth2.
    Usa as variáveis de ambiente:
        - OPENSKY_USER
    """
    global _conn, _cur

    if _conn is not None and _cur is not None:
        return _conn, _cur

    user = os.getenv("OPENSKY_USER")
    if not user:
        raise RuntimeError("Por favor defina OPENSKY_USER no ambiente")

    try:
        auth = OAuth2Authentication(redirect_auth_url_handler=ConsoleRedirectHandler())
        _conn = trino.dbapi.connect(
            host="trino.opensky-network.org",
            port=443,
            user=user,
            http_scheme="https",
            auth=auth,
            catalog="minio",
            schema="osky"
        )
        _cur = _conn.cursor()
        logger.info("Conexão com OpenSky estabelecida com sucesso")
        return _conn, _cur

    except Exception as e:
        logger.exception("Erro ao conectar no OpenSky/Trino")
        raise

def get_cursor():
    """
    Retorna apenas o cursor, garantindo que a conexão exista.
    """
    _, cur = create_connection()
    return cur
