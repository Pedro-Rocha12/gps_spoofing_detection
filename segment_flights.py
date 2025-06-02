# segment_flights.py
import pandas as pd

def segmentar_voos_por_icao24(track_csv: str, 
                              out_csv: str = None,
                              gap_threshold: int = 4200):
    """
    1) Carrega CSV de track com colunas [icao24, callsign, time, lat, lon, baroaltitude, heading, onground]
    2) Para cada icao24, calcula diff de tempo; sempre que diff > gap_threshold (segundos),
       ou quando detecta que a aeronave pousou e ficou em solo (onground=True) 
       por um intervalo, começamos novo 'flight_id'.
    3) Gera coluna 'flight_id' no formato "<icao24>_<n>".
    4) Salva o DataFrame resultante num CSV.
    """

    print("🔄 Lendo arquivo de track …")
    df = pd.read_csv(track_csv)
    df = df.sort_values(["icao24", "time"]).reset_index(drop=True)

    df["prev_time"] = df.groupby("icao24")["time"].shift(1)
    df["time_diff"] = df["time"] - df["prev_time"]
    df["is_new_segment"] = False

    df.loc[df["prev_time"].isna(), "is_new_segment"] = True
    df.loc[df["time_diff"] > gap_threshold, "is_new_segment"] = True

    df["segment_counter"] = df.groupby("icao24")["is_new_segment"].cumsum()
    df["flight_id"] = df["icao24"] + "_" + df["segment_counter"].astype(int).astype(str)

    df = df.drop(columns=["prev_time", "time_diff", "is_new_segment", "segment_counter"])

    if out_csv:
        print(f"✅ Salvando CSV segmentado em {out_csv} …")
        df.to_csv(out_csv, index=False)

    return df


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python segment_flights.py <caminho_para_track_csv> [<caminho_saida_csv>]")
        sys.exit(1)

    track_path = sys.argv[1]
    out_path = None
    if len(sys.argv) >= 3:
        out_path = sys.argv[2]

    df_segmentado = segmentar_voos_por_icao24(track_path, out_csv=out_path)
    print("🎉 Segmentação concluída. Exemplo de registros segmentados:")
    print(df_segmentado.head(10).to_string(index=False))
