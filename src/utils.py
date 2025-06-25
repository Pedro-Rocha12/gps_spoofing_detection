# src/utils.py

import os
import glob
import json
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Lista de nomes de features, na ordem das colunas de X
FEATURE_NAMES = [
    'lat',            # latitude (graus)
    'lon',            # longitude (graus)
    'geoaltitude',    # altitude geográfica (metros)
    'speed',          # velocidade (m/s)
    'accel',          # aceleração (m/s²)
    'turn_rate',      # taxa de curva (°/s)
    'vert_rate'       # taxa vertical (m/s)
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
    required = {'time', 'lat', 'lon', 'velocity', 'heading', 'vertrate', 'geoaltitude'}
    for p in paths:
        df = pd.read_csv(p)
        missing = required - set(df.columns)
        if missing:
            logger.warning("Pulando %s: faltam colunas %s", os.path.basename(p), missing)
            continue
        df = df.dropna(subset=required)
        if len(df) < 2:
            logger.warning("Pulando %s: menos de 2 pontos válidos após limpeza", os.path.basename(p))
            continue
        df['time'] = pd.to_datetime(df['time'].astype(int), unit='s', utc=True)
        dfs.append(df)
        names.append(os.path.basename(p))
    logger.info("Carregadas %d rotas válidas de %d arquivos", len(dfs), len(paths))
    return names, dfs


def extract_features(df: pd.DataFrame) -> np.ndarray:
    """
    Dado um DataFrame de rota, devolve X: array (n_samples, n_features)
    Features em cada instante t (a partir do segundo ponto):
      [ lat, lon, geoaltitude,
        speed, accel, turn_rate, vert_rate ]
    """
    lat = df['lat'].to_numpy()
    lon = df['lon'].to_numpy()
    alt = df['geoaltitude'].to_numpy()

    v = df['velocity'].to_numpy()
    t = df['time'].astype(np.int64) / 1e9

    dv = np.diff(v)
    dt = np.diff(t)
    dt[dt == 0] = 1e-6  # evitar divisão por zero

    accel = dv / dt

    hdg = df['heading'].to_numpy()
    dh = np.abs((hdg[1:] - hdg[:-1] + 180) % 360 - 180)
    turn_rate = dh / dt

    vert_rate = df['vertrate'].to_numpy()[1:]

    X = np.stack([
        lat[1:], lon[1:], alt[1:],
        v[1:], accel, turn_rate, vert_rate
    ], axis=1)
    return X


def build_and_split(
    tracks_dir: str,
    out_dir: str = "data/raw/tracks",
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42
):
    """
    1) Carrega as rotas brutas de tracks_dir e extrai features via extract_features.
    2) Concatena em X_all e routes_all.
    3) Separa estratificadamente em train/val/test:
         - test_size na primeira divisão
         - val_size/(1-test_size) na segunda divisão
    4) Ajusta StandardScaler no X_train, aplica em X_train, X_val, X_test.
    5) Salva em out_dir:
       - train.npz, val.npz, test.npz
       - scaler.pkl
       - feature_names.json
       - summary_split.json (contendo contagens de rotas e pontos)
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

    # 1ª divisão: treino+val vs test
    X_temp, X_test, routes_temp, routes_test = train_test_split(
        X_all, routes_all,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    # 2ª divisão: treino vs val
    val_frac = val_size / (1 - test_size)
    X_train, X_val, routes_train, routes_val = train_test_split(
        X_temp, routes_temp,
        test_size=val_frac,
        random_state=random_state,
        shuffle=True
    )

    # normalização
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    # salvar artefatos
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "train": os.path.join(out_dir, "train.npz"),
        "val":   os.path.join(out_dir, "val.npz"),
        "test":  os.path.join(out_dir, "test.npz"),
        "scaler": os.path.join(out_dir, "scaler.pkl"),
        "features": os.path.join(out_dir, "feature_names.json"),
        "summary": os.path.join(out_dir, "summary_split.json")
    }

    np.savez(paths["train"], X=X_train_scaled, routes=routes_train)
    np.savez(paths["val"],   X=X_val_scaled,   routes=routes_val)
    np.savez(paths["test"],  X=X_test_scaled,  routes=routes_test)
    joblib.dump(scaler, paths["scaler"])
    with open(paths["features"], 'w') as f:
        json.dump(FEATURE_NAMES, f, indent=2)

    # criar resumo das partições
    summary = {
        "n_routes_total": len(names),
        "n_points_total": int(X_all.shape[0]),
        "train": {
            "n_points": int(X_train_scaled.shape[0]),
            "n_routes": int(pd.Series(routes_train).nunique())
        },
        "val": {
            "n_points": int(X_val_scaled.shape[0]),
            "n_routes": int(pd.Series(routes_val).nunique())
        },
        "test": {
            "n_points": int(X_test_scaled.shape[0]),
            "n_routes": int(pd.Series(routes_test).nunique())
        }
    }
    with open(paths["summary"], 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("✔️ Train data: %s", paths["train"])
    logger.info("✔️ Val   data: %s", paths["val"])
    logger.info("✔️ Test  data: %s", paths["test"])
    logger.info("✔️ Scaler:    %s", paths["scaler"])
    logger.info("✔️ Features:  %s", paths["features"])
    logger.info("✔️ Summary:   %s", paths["summary"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extrai features de statevectors, normaliza, divide em train/val/test e salva artefatos."
    )
    parser.add_argument(
        "--tracks_dir",
        default="data/tracks",
        help="Diretório de entrada com statevectors_*.csv (padrão: data/tracks)"
    )
    parser.add_argument(
        "--out_dir",
        default="data/raw/tracks",
        help="Pasta de saída para train.npz, val.npz, test.npz, scaler.pkl, feature_names.json e summary_split.json"
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.1,
        help="Proporção de validação (padrão: 0.1 = 10%%)"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="Proporção de teste (padrão: 0.1 = 10%%)"
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
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_state
    )
