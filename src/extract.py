# src/extract.py
import pandas as pd

def load_track(csv_path: str) -> pd.DataFrame:
    """
    Lê o CSV de track e converte o timestamp.
    Espera colunas: icao24,callsign,time,lat,lon,baroaltitude,heading,onground
    """
    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['time'], unit='s')
    # opcional: filtrar apenas voos que importam
    return df
