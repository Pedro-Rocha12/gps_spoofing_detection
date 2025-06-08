# src/utils.py

import os
import glob
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Lista de nomes de features, na ordem das colunas de X
FEATURE_NAMES = [
    'lat',         # latitude (graus)
    'lon',         # longitude (graus)
    'alt',         # altitude (metros)
    'speed',       # velocidade (m/s)
    'accel',       # aceleração (m/s²)
    'turn_rate',   # taxa de curva (°/s)
    'vert_rate'    # taxa vertical (m/s)
]


def load_tracks(tracks_dir):
    """
    Itera sobre os CSVs em tracks_dir, faz drop de NaNs essenciais
    e retorna duas listas paralelas:
      - names: nome de cada arquivo (string)
      - dfs:   DataFrame correspondente, com colunas time, lat, lon,
               velocity, heading, vertrate, geoaltitude
    """
    paths = glob.glob(os.path.join(tracks_dir, "statevectors_*.csv"))
    dfs, names = [], []
    required = {'time','lat','lon','velocity','heading','vertrate','geoaltitude'}
    for p in paths:
        df = pd.read_csv(p)
        if required.issubset(df.columns):
            df = df.dropna(subset=required)
            if len(df) >= 2:
                df['time'] = pd.to_datetime(df['time'].astype(int), unit='s', utc=True)
                dfs.append(df)
                names.append(os.path.basename(p))
    return names, dfs


def extract_features(df):
    """
    Dado um DataFrame de rota, devolve X: array (n_samples, n_features)
    Features em cada instante t (a partir do segundo ponto):
      [ lat, lon, altitude,
        speed, accel, turn_rate, vert_rate ]
    onde:
      - lat, lon       em graus
      - altitude       em metros
      - speed          em m/s
      - accel          em m/s²
      - turn_rate      em °/s
      - vert_rate      em m/s
    """
    # coordenadas e altitude
    lat = df['lat'].to_numpy()
    lon = df['lon'].to_numpy()
    alt = df['geoaltitude'].to_numpy()

    # velocidade e tempo
    v = df['velocity'].to_numpy()
    t = df['time'].astype(np.int64) / 1e9

    # diffs de velocidade e tempo
    dv = np.diff(v)
    dt = np.diff(t)
    dt[dt == 0] = 1e-6

    accel = dv / dt

    # turn rate
    hdg = df['heading'].to_numpy()
    dh = np.abs((hdg[1:] - hdg[:-1] + 180) % 360 - 180)
    turn_rate = dh / dt

    # vertical rate
    vert_rate = df['vertrate'].to_numpy()[1:]

    # empacota todas as features alinhadas em len(dt)
    X = np.stack([
        lat[1:], lon[1:], alt[1:],
        v[1:], accel, turn_rate, vert_rate
    ], axis=1)
    return X


def build_and_split(tracks_dir,
                    out_dir="data/tracks",
                    test_size=0.1,
                    random_state=42):
    """
    1) Carrega as rotas brutas de tracks_dir e extrai features via extract_features.
    2) Concatena em X_all (shape [total_points, n_features]) e routes_all (shape [total_points]).
    3) Separa aleatoriamente em treino/val (proporção test_size).
    4) Ajusta StandardScaler no X_train, aplica em X_train e X_val.
    5) Salva em out_dir:
       - train.npz  (X=X_train_scaled, routes=routes_train)
       - val.npz    (X=X_val_scaled,   routes=routes_val)
       - scaler.pkl (o StandardScaler ajustado)
       - feature_names.json (lista FEATURE_NAMES)
    """
    names, dfs = load_tracks(tracks_dir)
    if not dfs:
        raise RuntimeError(f"Nenhuma rota válida encontrada em {tracks_dir}")

    # extrai X e mantém rota associada a cada ponto
    X_list, routes_list = [], []
    for name, df in zip(names, dfs):
        X = extract_features(df)
        X_list.append(X)
        routes_list.extend([name] * X.shape[0])

    X_all = np.vstack(X_list)
    routes_all = np.array(routes_list)

    # split treino/val
    X_train, X_val, routes_train, routes_val = train_test_split(
        X_all, routes_all,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    # normalização
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled   = scaler.transform(X_val)

    # criar saída
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.npz")
    val_path   = os.path.join(out_dir,   "val.npz")
    scaler_path = os.path.join(out_dir, "scaler.pkl")
    features_path = os.path.join(out_dir, "feature_names.json")

    # salvar NPZ
    np.savez(train_path, X=X_train_scaled, routes=routes_train)
    np.savez(val_path,   X=X_val_scaled,   routes=routes_val)

    # salvar scaler e nomes de feature
    joblib.dump(scaler, scaler_path)
    with open(features_path, 'w') as f:
        json.dump(FEATURE_NAMES, f)

    print(f"✔️ Train data: {train_path}")
    print(f"✔️ Val   data: {val_path}")
    print(f"✔️ Scaler:    {scaler_path}")
    print(f"✔️ Features:  {features_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extrai features de statevectors, normaliza, divide em train/val e salva artefatos."
    )
    parser.add_argument(
        "--tracks_dir",
        default="data/tracks",
        help="(opcional) Diretório de entrada com statevectors_*.csv (padrão: data/tracks)"
    )
    parser.add_argument(
        "--out_dir",
        default="data/raw/tracks",
        help="Pasta de saída para train.npz, val.npz, scaler.pkl e feature_names.json (padrão: data/raw/tracks)"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="Proporção de validação (padrão: 0.1 = 10%)"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Seed para separação aleatória (padrão: 42)"
    )
    args = parser.parse_args()

    build_and_split(
        tracks_dir=args.tracks_dir,
        out_dir=args.out_dir,
        test_size=args.test_size,
        random_state=args.random_state
    )
