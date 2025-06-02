# extract.py

import os
import logging
import pandas as pd
from connect_opensky import get_cursor
from queries_opensky import (
    query_flights_between,
    query_unpacked_track
)

logger = logging.getLogger(__name__)

def extract(
    origin: str,
    destination: str,
    start_ts: int,
    end_ts: int,
    limit_flights: int = 3000,
    output_dir: str = "output",
    flights_filename: str = None,
    track_filename: str = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extrai e salva:
      - metadados dos voos origin→destination no intervalo [start_ts, end_ts)
      - histórico de track desses voos (desempacotado)

    Parâmetros:
      origin, destination: ICAO de aeroportos (e.g. 'SBSP','SBBR')
      start_ts, end_ts: timestamps Unix para filtro de time window
      limit_flights: máximo de voos a buscar
      output_dir: pasta onde salvar os CSVs
      flights_filename: nome do CSV de metadados (padrão gerado automaticamente)
      track_filename: nome do CSV de track (idem)

    Retorna:
      (df_flights, df_track)
    """
    cur = get_cursor()
    os.makedirs(output_dir, exist_ok=True)

    # Definir nomes de arquivo padrão se não fornecidos
    if flights_filename is None:
        flights_filename = f"voos_{origin.lower()}_{destination.lower()}_{start_ts}_{end_ts}.csv"
    if track_filename is None:
        track_filename = f"track_{origin.lower()}_{destination.lower()}_{start_ts}_{end_ts}.csv"

    # 1️⃣ Metadados de voo
    try:
        logger.info("Buscando metadados dos voos %s→%s em %d–%d", origin, destination, start_ts, end_ts)
        sql_flights = query_flights_between(
            origin, destination, start_ts, end_ts, limit=limit_flights
        )
        cur.execute(sql_flights)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        df_flights = pd.DataFrame(rows, columns=cols)

        if df_flights.empty:
            logger.warning("Nenhum voo encontrado para %s→%s nesse período", origin, destination)
            return pd.DataFrame(), pd.DataFrame()

        path_f = os.path.join(output_dir, flights_filename)
        df_flights.to_csv(path_f, index=False)
        logger.info("Metadados salvos em %s", path_f)

    except Exception:
        logger.exception("Falha ao extrair metadados de voo")
        raise

    # 2️⃣ Track (desempacotado)
    try:
        actual_limit = len(df_flights)
        logger.info("Extraindo track para %d voos em única query", actual_limit)
        sql_track = query_unpacked_track(
            origin, destination, start_ts, end_ts, limit_flights=actual_limit
        )
        cur.execute(sql_track)
        rows2 = cur.fetchall()
        cols2 = [d[0] for d in cur.description]
        df_track = pd.DataFrame(rows2, columns=cols2)

        if df_track.empty:
            logger.warning("Nenhum track encontrado para esses voos")
            return df_flights, pd.DataFrame()

        path_t = os.path.join(output_dir, track_filename)
        df_track.to_csv(path_t, index=False)
        logger.info("Track salvo em %s", path_t)

    except Exception:
        logger.exception("Falha ao extrair track")
        raise

    return df_flights, df_track
