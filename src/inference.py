#!/usr/bin/env python3
# src/inference.py

import os
import json
import joblib
import argparse
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from src.utils import extract_features

def load_models(args):
    """Carrega scaler e os três detectores."""
    scaler = joblib.load(args.scaler)
    iso    = joblib.load(args.iforest)
    oc     = joblib.load(args.ocsvm)
    ae     = load_model(args.autoencoder)
    with open(args.ae_threshold) as f:
        ae_thr = json.load(f)["threshold"]
    return scaler, iso, oc, ae, ae_thr

def run_inference(track_csv, scaler, iso, oc, ae, ae_thr):
    """Extrai features de um CSV, normaliza e retorna scores/preds para cada modelo."""
    df = pd.read_csv(track_csv)
    df = df.dropna(subset=['time','lat','lon','velocity','heading','vertrate','geoaltitude'])
    df['time'] = pd.to_datetime(df['time'].astype(int), unit='s', utc=True)
    X = extract_features(df)
    Xs = scaler.transform(X)

    # IsolationForest
    score_iso = -iso.decision_function(Xs)
    pred_iso  = (iso.predict(Xs) == -1).astype(int)

    # One-Class SVM
    score_oc = -oc.score_samples(Xs)
    pred_oc  = (oc.predict(Xs) == -1).astype(int)

    # Autoencoder
    recon     = ae.predict(Xs)
    mse       = np.mean((recon - Xs)**2, axis=1)
    score_ae  = mse
    pred_ae   = (mse > ae_thr).astype(int)

    return df, {
        "iforest":   (score_iso, pred_iso),
        "ocsvm":     (score_oc, pred_oc),
        "autoencoder": (score_ae, pred_ae),
    }

def save_results(df, results, out_dir, basename):
    """Salva CSV com time, lat/lon originais + scores/preds de cada modelo."""
    os.makedirs(out_dir, exist_ok=True)
    out = df.iloc[1:].copy()  # extrai correspondentes a cada linha de X
    for name, (scores, preds) in results.items():
        out[f"{name}_score"] = scores
        out[f"{name}_anom"]  = preds
    path = os.path.join(out_dir, basename + "_inference.csv")
    out.to_csv(path, index=False)
    print(f"✅ Resultados salvos em {path}")

def main():
    p = argparse.ArgumentParser(description="Inference para detecção de spoofing")
    p.add_argument("track_csv", help="CSV de state vectors de uma única rota")
    p.add_argument("--scaler",      required=True, help="caminho para scaler.pkl")
    p.add_argument("--iforest",     required=True, help="caminho para iso_forest.pkl")
    p.add_argument("--ocsvm",       required=True, help="caminho para oc_svm.pkl")
    p.add_argument("--autoencoder", required=True, help="caminho para autoencoder.h5")
    p.add_argument("--ae_threshold",required=True, help="caminho para threshold.json do AE")
    p.add_argument("--out_dir",     default="inference_results",
                   help="pasta para salvar o CSV de saída")
    args = p.parse_args()

    scaler, iso, oc, ae, ae_thr = load_models(args)
    df, results = run_inference(args.track_csv, scaler, iso, oc, ae, ae_thr)
    base = os.path.splitext(os.path.basename(args.track_csv))[0]
    save_results(df, results, args.out_dir, base)

if __name__ == "__main__":
    main()
