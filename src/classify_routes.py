# classify_routes.py

import os, glob
import joblib
import numpy as np
import pandas as pd
from utils import extract_features

def load_model_and_scaler(iforest_path, scaler_path):
    iso   = joblib.load(iforest_path)
    scaler = joblib.load(scaler_path)
    return iso, scaler

def score_route(df, iso, scaler):
    X = extract_features(df)
    Xs = scaler.transform(X)
    preds = iso.predict(Xs)
    return preds

def classify_all_routes(tracks_dir, iso, scaler, anomaly_frac_thresh=0.05):
    records = []
    pattern = os.path.join(tracks_dir, "*.csv")
    for fp in glob.glob(pattern):
        name = os.path.basename(fp)
        df = pd.read_csv(fp).dropna(subset=['lat','lon','velocity','heading','vertrate','geoaltitude','time'])
        if len(df) < 2:
            continue
        preds = score_route(df, iso, scaler)
        n_pts   = len(preds)
        n_anom  = int((preds == -1).sum())
        frac    = n_anom / n_pts
        is_spoof = frac >= anomaly_frac_thresh
        records.append({
            'route': name,
            'n_points': n_pts,
            'n_anomalous': n_anom,
            'frac_anomalous': frac,
            'is_spoof': is_spoof
        })
    return pd.DataFrame(records)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Classifica rotas inteiras como spoofed segundo IsolationForest"
    )
    parser.add_argument("tracks_dir",
                        help="Pasta com CSVs de rota (statevectors_*.csv)")
    parser.add_argument("--model",  default="models/iforest/iso_forest.pkl")
    parser.add_argument("--scaler", default="data/raw/tracks/scaler.pkl")
    parser.add_argument("--thresh", type=float, default=0.05,
                        help="Fração mínima de pontos anômalos para marcar rota")
    parser.add_argument("--out_csv", default="route_classification.csv")
    args = parser.parse_args()

    iso, scaler = load_model_and_scaler(args.model, args.scaler)
    df_res = classify_all_routes(
        args.tracks_dir, iso, scaler, anomaly_frac_thresh=args.thresh
    )
    df_res.to_csv(args.out_csv, index=False)
    spoofed = df_res[df_res.is_spoof]
    print(f"Total rotas: {len(df_res)}, spoofed (≥{args.thresh*100:.1f}% pontos): {len(spoofed)}")
    print("Rotas marcadas como spoofed:")
    print(spoofed['route'].to_list())
