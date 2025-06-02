#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import logging
import pandas as pd
import multiprocessing as mp
import signal
import sys
from typing import Dict

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from connect_opensky import get_cursor
from queries_opensky import query_state_vectors_for_flight

# Configuração básica de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _processar_um_voo(flight: Dict[str, str], tracks_dir: str) -> None:
    """
    Função executada no processo filho para extrair state vectors de um voo.

    Parâmetros:
        flight: dicionário com as chaves:
            "icao24", "callsign", "estdepartureairport", "estarrivalairport", 
            "firstseen", "lastseen"
        tracks_dir: diretório onde será salvo o CSV resultante de state vectors.
    Comportamento:
        - Conecta ao Trino (via get_cursor)
        - Monta a SQL usando query_state_vectors_for_flight
        - Executa a query, bota o resultado em DataFrame, salva em CSV dentro de tracks_dir
        - Em caso de exceção, interceta-a, remove qualquer CSV parcial e relança para exitcode != 0
    """
    icao24 = flight["icao24"]
    callsign = flight["callsign"].strip()
    dep = flight["estdepartureairport"]
    arr = flight["estarrivalairport"]
    firstseen = int(flight["firstseen"])
    lastseen = int(flight["lastseen"])

    # Nome padrão para o CSV de state vectors deste voo
    filename = f"statevectors_{icao24}_{firstseen}_{lastseen}.csv"
    output_path = os.path.join(tracks_dir, filename)

    try:
        # 1) Obter cursor Trino
        cur = get_cursor()

        # 2) Gerar SQL
        sql = query_state_vectors_for_flight(icao24, firstseen, lastseen)

        # 3) Executar
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]

        # 4) Construir DataFrame e salvar em CSV
        df = pd.DataFrame(rows, columns=cols)
        df.to_csv(output_path, index=False)

        logger.info("✔️  Voo %s: state vectors salvos em %s", icao24, output_path)

    except Exception:
        # Se houve qualquer erro, remover CSV parcial (se existir) e relançar exceção para exitcode != 0
        logger.exception("❌  Voo %s: falha ao extrair state vectors", icao24)
        if os.path.exists(output_path):
            os.remove(output_path)
        # relança para que o processo filho termine com código != 0
        raise


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Pega um CSV de voos (colunas: icao24,callsign,estdepartureairport,"
            "estarrivalairport,firstseen,lastseen,...) e, para cada linha, executa "
            "query_state_vectors_for_flight. Mantém três CSVs de controle: "
            "  • pendentes (o próprio CSV de entrada, que vai sendo reduzido)  "
            "  • completados  "
            "  • falhados"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "input_csv",
        type=str,
        help=(
            "Caminho para o CSV de entrada contendo as colunas:\n"
            "  icao24,callsign,estdepartureairport,estarrivalairport,firstseen,lastseen,..."
        ),
    )
    parser.add_argument(
        "--tracks_dir",
        type=str,
        default="tracks",
        help="Diretório onde serão gravados os CSVs de state vectors (padrão: tracks/)",
    )
    parser.add_argument(
        "--completed_csv",
        type=str,
        default="completed.csv",
        help="CSV onde serão acumulados voos processados com sucesso (padrão: completed.csv)",
    )
    parser.add_argument(
        "--failed_csv",
        type=str,
        default="failed.csv",
        help="CSV onde serão acumulados voos que falharam (padrão: failed.csv)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Tempo máximo (em segundos) por consulta (padrão: 1800 = 30 min)",
    )
    args = parser.parse_args()

    # 0) Verificações iniciais
    if not os.path.exists(args.input_csv):
        logger.error("CSV de entrada não encontrado: %s", args.input_csv)
        sys.exit(1)

    # Certificar diretório de saída de tracks
    os.makedirs(args.tracks_dir, exist_ok=True)

    # 1) Carregar CSV de entrada (pendentes) numa lista de dicionários
    df_input = pd.read_csv(args.input_csv, dtype=str)
    # Garantir que colunas mínimas existam
    col_minimas = {
        "icao24",
        "callsign",
        "estdepartureairport",
        "estarrivalairport",
        "firstseen",
        "lastseen",
    }
    if not col_minimas.issubset(df_input.columns):
        logger.error(
            "O CSV de entrada deve conter pelo menos as colunas: %s", ", ".join(col_minimas)
        )
        sys.exit(1)

    # Converter cada linha em um dicionário (para facilitar passagem ao processo filho)
    lista_voos = df_input.to_dict(orient="records")

    # 2) Se já existir um CSV de completed, carregue; caso contrário, crie vazio
    if os.path.exists(args.completed_csv):
        df_completed = pd.read_csv(args.completed_csv, dtype=str)
    else:
        df_completed = pd.DataFrame(columns=df_input.columns)

    # 3) Se já existir um CSV de failed, carregue; caso contrário, crie vazio
    if os.path.exists(args.failed_csv):
        df_failed = pd.read_csv(args.failed_csv, dtype=str)
    else:
        df_failed = pd.DataFrame(columns=df_input.columns)

    logger.info("Iniciando processamento de %d voos pendentes...", len(lista_voos))

    try:
        # 4) Iterar sobre cada voo na lista
        for i, voo in enumerate(lista_voos):
            icao24 = voo["icao24"]
            firstseen = voo["firstseen"]
            lastseen = voo["lastseen"]
            logger.info(
                "[%d/%d] Processando voo icao24=%s (firstseen=%s, lastseen=%s)...",
                i + 1,
                len(lista_voos),
                icao24,
                firstseen,
                lastseen,
            )

            # 4.1) Montar e iniciar um processo filho para executar o _processar_um_voo
            p = mp.Process(target=_processar_um_voo, args=(voo, args.tracks_dir))
            p.start()

            # 4.2) Aguardar até timeout (em segundos). Se passar de timeout, encerraremos o processo.
            p.join(timeout=args.timeout)
            if p.is_alive():
                # Ainda está rodando após timeout → consideramos “falha por timeout”
                p.terminate()
                p.join()
                logger.warning(
                    "⏱  Voo %s: tempo limite de %d s atingido. Marcado como falha.",
                    icao24,
                    args.timeout,
                )
                # Adiciona ao DataFrame de falhados
                df_failed = pd.concat([df_failed, pd.DataFrame([voo])], ignore_index=True)

            else:
                # Se não está mais vivo, verificar o exitcode
                if p.exitcode == 0:
                    logger.info("✅  Voo %s: processado com sucesso.", icao24)
                    df_completed = pd.concat([df_completed, pd.DataFrame([voo])], ignore_index=True)
                else:
                    logger.warning("❌  Voo %s: processo filho falhou (exitcode=%d).", icao24, p.exitcode)
                    df_failed = pd.concat([df_failed, pd.DataFrame([voo])], ignore_index=True)

            # 4.3) Remover este voo do CSV de entrada (pendentes) em disco
            try:
                # Recarrega o CSV de pendentes (caso tenha sido atualizado em execuções anteriores)
                df_atual = pd.read_csv(args.input_csv, dtype=str)
            except Exception:
                df_atual = pd.DataFrame(columns=df_input.columns)

            # Filtrar todas as linhas que NÃO sejam exatamente este voo, considerando icao24+firstseen+lastseen
            cond_remover = ~(
                (df_atual["icao24"] == voo["icao24"])
                & (df_atual["firstseen"] == voo["firstseen"])
                & (df_atual["lastseen"] == voo["lastseen"])
            )
            df_atual = df_atual[cond_remover]
            # Salvar o CSV de pendentes atualizado
            df_atual.to_csv(args.input_csv, index=False)

            # 4.4) Atualizar e gravar CSVs de completed e failed (sempre sobrescrever)
            df_completed.to_csv(args.completed_csv, index=False)
            df_failed.to_csv(args.failed_csv, index=False)

            # Pequena pausa opcional (evita sobrecarga de disco em laços muito rápidos)
            # time.sleep(0.5)

        logger.info("🎉 Processamento concluído. Não há mais voos pendentes.")

    except KeyboardInterrupt:
        # Se o usuário apertar Ctrl+C, interrompemos a iteração imediatamente
        logger.info("⁉️  Interrompido pelo usuário. Salvando estado atual e saindo...")
        # Antes de sair, já salvamos os CSVs (pendentes, completed e failed)
        # Os pendentes são exatamente o que restou no arquivo args.input_csv,
        # pois já foi sendo atualizado a cada loop.
        # Os CSVs de completed e failed já foram salvos no loop “finally” abaixo.
        sys.exit(0)

    except Exception:
        # Qualquer outra exceção não prevista: logar e sair
        logger.exception("❌  Ocorreu um erro inesperado. Gravando estado atual e abortando.")
        sys.exit(1)


if __name__ == "__main__":
    # Para garantir que o Ctrl+C mate também os processos filhos imediatamente
    signal.signal(signal.SIGINT, signal.default_int_handler)
    main()
