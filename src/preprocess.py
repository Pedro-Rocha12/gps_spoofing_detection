# src/preprocess.py
import numpy as np
import pandas as pd

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera features por ponto de track:
     - Δ distância (Haversine)
     - Δ tempo (s)
     - velocidade instantânea (m/s)
     - aceleração (m/s²)
     - variação de heading (deg)
     - Δ altitude (m)
    """
    R = 6371e3

    def haversine(lat1, lon1, lat2, lon2):
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlmb = np.radians(lon2 - lon1)
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlmb/2)**2
        return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1-a))

    df = df.sort_values(['icao24','time']).reset_index(drop=True)

    # deslocamentos
    df['lat_prev'] = df.groupby('icao24')['lat'].shift()
    df['lon_prev'] = df.groupby('icao24')['lon'].shift()
    df['time_prev'] = df.groupby('icao24')['time'].shift()
    df['alt_prev'] = df.groupby('icao24')['baroaltitude'].shift()
    df['heading_prev'] = df.groupby('icao24')['heading'].shift()

    df['dist_m'] = haversine(df['lat_prev'], df['lon_prev'], df['lat'], df['lon'])
    df['dt']     = df['time'] - df['time_prev']
    df['speed'] = df['dist_m'] / df['dt'].replace(0, np.nan)    # m/s
    df['accel'] = df.groupby('icao24')['speed'].diff() / df['dt'].replace(0, np.nan)
    df['dheading'] = (df['heading'] - df['heading_prev']).abs()
    df['dalt']  = df['baroaltitude'] - df['alt_prev']

    # limpa pontos sem dt válido
    df = df.dropna(subset=['speed','accel','dheading','dalt'])

    return df[[
        'icao24','callsign','time','datetime',
        'lat','lon','baroaltitude','heading','onground',
        'dist_m','dt','speed','accel','dheading','dalt'
    ]]
