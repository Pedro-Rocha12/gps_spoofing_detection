# lookup.py

import argparse
import os
import pandas as pd
from connect_opensky import get_cursor
from queries_opensky import (
    query_detalhes_voo,
    query_state_vectors_for_flight,
    query_track_for_flight
)

def lookup_flight(args):
    cur = get_cursor()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) detalhes do voo (flights_data5)
    sql_f = query_detalhes_voo(
        args.icao24,
        args.callsign,
        args.dep,
        args.arr,
        args.firstseen,
        args.lastseen
    )
    cur.execute(sql_f)
    df_f = pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])
    out_f = os.path.join(
        args.output_dir,
        f"flight_{args.icao24}_{args.firstseen}_{args.lastseen}.csv"
    )
    df_f.to_csv(out_f, index=False)
    print(f"✅ Detalhes do voo salvos em: {out_f}")

    # 2) state_vectors (opcional)
    if args.lookup_type in ("state_vectors", "all"):
        sql_sv = query_state_vectors_for_flight(
            args.icao24,
            args.firstseen,
            args.lastseen
        )
        cur.execute(sql_sv)
        df_sv = pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])
        out_sv = os.path.join(args.output_dir, f"state_vectors_{args.icao24}.csv")
        df_sv.to_csv(out_sv, index=False)
        print(f"✅ state_vectors salvos em: {out_sv}")

    # 3) track (opcional)
    if args.lookup_type in ("track", "all"):
        # garante que callsign, dep e arr estejam presentes
        if not all([args.callsign, args.dep, args.arr]):
            print("Para --lookup_type track, forneça também --callsign, --dep e --arr")
        else:
            sql_tr = query_track_for_flight(
                args.icao24,
                args.callsign,
                args.dep,
                args.arr,
                args.firstseen,
                args.lastseen
            )
            cur.execute(sql_tr)
            df_tr = pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])
            out_tr = os.path.join(args.output_dir, f"track_{args.icao24}.csv")
            df_tr.to_csv(out_tr, index=False)
            print(f"✅ track salvos em: {out_tr}")


def main():
    parser = argparse.ArgumentParser(
        description="Lookup de voo individual na base OpenSky"
    )
    parser.add_argument("--icao24",    required=True, help="ICAO24 da aeronave")
    parser.add_argument("--callsign",  required=True, help="Callsign exato (sem espaços)")
    parser.add_argument("--dep",       required=True, help="ICAO do aeroporto de partida")
    parser.add_argument("--arr",       required=True, help="ICAO do aeroporto de chegada")
    parser.add_argument("--firstseen", type=int, required=True, help="Timestamp Unix firstseen")
    parser.add_argument("--lastseen",  type=int, required=True, help="Timestamp Unix lastseen")
    parser.add_argument(
        "--lookup_type",
        choices=["flights", "state_vectors", "track", "all"],
        default="all",
        help="Dados a buscar"
    )
    parser.add_argument(
        "--output_dir",
        default="output",
        help="Pasta onde salvar os CSVs"
    )
    args = parser.parse_args()
    lookup_flight(args)

if __name__ == "__main__":
    main()
