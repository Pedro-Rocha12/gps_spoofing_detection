import argparse
import os
import pandas as pd

from connect_opensky import get_cursor
from extract import extract
from visualize import plot_tracks
from explore import explore_to_csv
from lookup import lookup_flight
from queries_opensky import query_list_flights_by_icao


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline OpenSky: extração, exploração, lookup e visualização de trajetórias"
    )

    # parâmetros de rota
    parser.add_argument("--origin", type=str, default="SBSP",
                        help="ICAO do aeroporto de origem (ex: SBSP)")
    parser.add_argument("--destination", type=str, default="SBBR",
                        help="ICAO do aeroporto de destino (ex: SBBR)")
    parser.add_argument("--start_ts", type=int, default=1704067200,
                        help="Timestamp Unix início do período (inclusive)")
    parser.add_argument("--end_ts", type=int, default=1735689600,
                        help="Timestamp Unix fim do período (exclusive)")
    parser.add_argument("--limit", type=int, default=3000,
                        help="Número máximo de voos a extrair")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Diretório base para salvar CSVs")

    # modos de operação
    parser.add_argument("--explore", action="store_true",
                        help="Explorar estrutura do banco Trino/OpenSky e salvar em CSV")
    parser.add_argument("--list-flights", action="store_true",
                        help="Listar voos por ICAO24 no período")
    parser.add_argument("--extract", action="store_true",
                        help="Extrair dados de voos e track e salvar CSVs")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualizar trajetórias a partir do CSV de track")
    parser.add_argument("--lookup", action="store_true",
                        help="Fazer lookup de voo individual (usando lookup.py)")

    # parâmetros de list-flights
    parser.add_argument("--icao24", type=str,
                        help="ICAO24 da aeronave para list-flights e lookup")

    # parâmetros de visualize
    parser.add_argument("--path_csv", type=str, default="output/track_segmentado.csv",
                        help="Caminho para o CSV de track segmentado")
    parser.add_argument("--max_flights", type=int, default=5,
                        help="Número máximo de voos a visualizar")
    parser.add_argument("--flight_ids", nargs="*",
                        help="Lista de ICAO24 para visualizar")

    # flags pass-through para lookup.py
    parser.add_argument("--callsign",   type=str, help="Callsign para lookup")
    parser.add_argument("--dep",        type=str, help="ICAO partida para lookup")
    parser.add_argument("--arr",        type=str, help="ICAO chegada para lookup")
    parser.add_argument("--firstseen",  type=int, help="firstseen timestamp para lookup")
    parser.add_argument("--lastseen",   type=int, help="lastseen timestamp para lookup")
    parser.add_argument(
        "--lookup_type",
        choices=["flights", "state_vectors", "track", "all"],
        default="flights",
        help="O que buscar no lookup"
    )

    args = parser.parse_args()

    # 1) explore
    if args.explore:
        os.makedirs("dados", exist_ok=True)
        explore_to_csv(output_path="dados/estrutura_opensky.csv")
        return

    # 2) list flights by icao24
    if args.list_flights:
        if not args.icao24:
            print("Para --list-flights, forneça --icao24!")
            return
        cur = get_cursor()
        sql = query_list_flights_by_icao(
            args.icao24,
            args.start_ts,
            args.end_ts,
            limit=10
        )
        cur.execute(sql)
        df_list = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
        if df_list.empty:
            print("Nenhum voo encontrado para esse ICAO24 e período.")
        else:
            print("Voos encontrados:\n", df_list.to_string(index=False))
        return

    # 3) lookup individual flight (delegado ao lookup.py)
    if args.lookup:
        # apenas repassamos todos os args para lookup_flight
        lookup_flight(args)
        return

    # 4) extract
    if args.extract:
        extract(
            origin=args.origin,
            destination=args.destination,
            start_ts=args.start_ts,
            end_ts=args.end_ts,
            limit_flights=args.limit,
            output_dir=args.output_dir
        )

    # 5) visualize
    if args.visualize:

        if not os.path.exists(args.path_csv):
            print(f"⚠️ CSV de track não encontrado: {args.path_csv}")
            return
        track_csv = args.path_csv
        # track_csv = os.path.join(
        #     args.output_dir,
        #     f"track_{args.origin.lower()}_{args.destination.lower()}_{args.start_ts}_{args.end_ts}.csv"
        # )
        # if not os.path.exists(track_csv):
        #     print("⚠️ Track CSV não encontrado. Rode com --extract primeiro.")
        #     return
        plot_tracks(
            csv_path=track_csv,
            flight_ids=args.flight_ids,
            max_flights=args.max_flights
        )


if __name__ == "__main__":
    main()
