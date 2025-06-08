#!/usr/bin/env python3
# evaluate_models.py

import os
import glob
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    classification_report
)
from src.utils import extract_features

def load_test_data(good_dir, bad_dir, scaler):
    Xs, ys = [], []
    for d, label in [(good_dir, 0), (bad_dir, 1)]:
        for fp in glob.glob(os.path.join(d, "statevectors_*.csv")):
            df = pd.read_csv(fp)
            # converter time, dropnas idem utils.load_tracks
            df = df.dropna(subset=['time','lat','lon','velocity','heading','vertrate','geoaltitude'])
            df['time'] = pd.to_datetime(df['time'].astype(int), unit='s', utc=True)
            if len(df) < 2: 
                continue
            X = extract_features(df)             # shape (n_pts, 7)
            X_scaled = scaler.transform(X)       # mesmo scaler do build_and_split
            Xs.append(X_scaled)
            ys.extend([label]*len(X_scaled))
    X_test = np.vstack(Xs)
    y_test = np.array(ys)
    return X_test, y_test

def load_models():
    # scaler
    scaler = joblib.load("data/raw/tracks/scaler.pkl")
    # IsolationForest
    iso = joblib.load("models/iforest/iso_forest.pkl")
    # OneClassSVM
    oc = joblib.load("models/ocsvm/oc_svm.pkl")
    # Autoencoder + threshold
    ae = load_model("models/autoencoder/autoencoder.h5")
    with open("models/autoencoder/threshold.json") as f:
        threshold = json.load(f)["threshold"]
    return scaler, iso, oc, ae, threshold

def get_scores(iso, oc, ae, threshold, X):
    # IsolationForest: invertendo decision_function
    scores_iso = -iso.decision_function(X)
    preds_iso  = iso.predict(X)==-1

    # OneClassSVM
    scores_oc  = -oc.decision_function(X)
    preds_oc   = oc.predict(X)==-1

    # Autoencoder: MSE e preds
    recon = ae.predict(X)
    mse   = np.mean((recon - X)**2, axis=1)
    scores_ae = mse
    preds_ae  = mse > threshold

    return (scores_iso, preds_iso,
            scores_oc,  preds_oc,
            scores_ae,  preds_ae)

def plot_roc_pr(y_test, scores, labels, out_dir):
    """Plota ROC e PR curves para cada modelo."""
    os.makedirs(out_dir, exist_ok=True)

    # ROC
    plt.figure()
    for score, lab in zip(scores, labels):
        fpr, tpr, _ = roc_curve(y_test, score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{lab} (AUC={roc_auc:.3f})")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"roc_comparison.png"))
    plt.close()

    # Precision-Recall
    plt.figure()
    for score, lab in zip(scores, labels):
        prec, rec, _ = precision_recall_curve(y_test, score)
        pr_auc = auc(rec, prec)
        plt.plot(rec, prec, label=f"{lab} (AUPR={pr_auc:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"pr_comparison.png"))
    plt.close()

def main():
    # 1) Carrega modelos e scaler
    scaler, iso, oc, ae, threshold = load_models()

    # 2) Carrega dados de teste
    X_test, y_test = load_test_data(
        good_dir="tracks_filtradas",
        bad_dir ="tracks_excluidas",
        scaler=scaler
    )
    print(f"Conjunto de teste: {len(y_test)} pontos, spoofed={y_test.sum()}, legit={len(y_test)-y_test.sum()}")

    # 3) Obtém scores e preds
    s_iso, p_iso, s_oc, p_oc, s_ae, p_ae = get_scores(iso, oc, ae, threshold, X_test)

    # 4) Plota ROC e PR
    plot_roc_pr(
        y_test,
        scores=[s_iso, s_oc, s_ae],
        labels=["IsolationForest","OC-SVM","Autoencoder"],
        out_dir="evaluation_plots"
    )
    print("✔️ Gráficos ROC e PR salvos em evaluation_plots/")

    # 5) Relatórios de classificação (usando preds)
    print("\n=== Classification Reports ===\n")
    for preds, name in zip([p_iso,p_oc,p_ae], ["IForest","OC-SVM","Autoenc"]):
        print(f"--- {name} ---")
        print(classification_report(y_test, preds, digits=3))

if __name__=="__main__":
    main()
