# src/main.py
import argparse, os
import pandas as pd
import numpy as np

from extract import load_track
from preprocess import preprocess
from models import (
    get_lof_model,
    train_isolation_forest,
    train_autoencoder,
    score_iso, score_lof, score_ae
)
from visualize import plot_anomalies

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--track_csv', required=True, help='path para track_full.csv')
    p.add_argument('--model', choices=['iso','lof','ae'], default='iso')
    p.add_argument('--output_dir', default='output')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    # 1) carga
    df = load_track(args.track_csv)
    # 2) preprocess
    df_feat = preprocess(df)
    X = df_feat[['speed','accel','dheading','dalt']].values

    # 3) treinar e pontuar
    if args.model == 'iso':
        scaler, model = train_isolation_forest(X)
        df_feat['score'] = score_iso(scaler, model, X)
        thr = np.percentile(df_feat['score'], 1)   # top 1% como anomalia
    elif args.model == 'lof':
        scaler, model = get_lof_model(X, n_neighbors=20)
        df_feat['score'] = score_lof(scaler, model, X)
        thr = np.percentile(df_feat['score'], 1)
    else:  # autoencoder
        scaler, model = train_autoencoder(X, epochs=20)
        df_feat['score'] = score_ae(scaler, model, X)
        thr = np.percentile(df_feat['score'], 99)  # erro alto = anomalia

    # 4) salvar resultados
    out_csv = os.path.join(args.output_dir, f'results_{args.model}.csv')
    df_feat.to_csv(out_csv, index=False)
    print(f"✅ Features + scores salvos em {out_csv}\nThreshold = {thr:.3f}")

    # 5) visualizar (primeiro voo só p/ demo)
    sample = df_feat[df_feat['icao24']==df_feat['icao24'].iloc[0]]
    plot_anomalies(sample, 'score', threshold=thr)

if __name__=='__main__':
    main()
