#!/usr/bin/env python3
# evaluate_models.py

import os
import glob
import json
import logging
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    classification_report
)
import joblib

from utils import extract_features

# --- Logging setup ---
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("evaluate_models")


def load_test_data(test_dir: str, spoofed_dir: str, scaler):
    """
    Carrega todos os CSVs de test_dir, extrai features e rotula cada ponto:
      • label=1 se o nome do arquivo estiver em spoofed_dir
      • label=0 caso contrário
    Retorna X_test (n_samples×n_features) e y_test (n_samples,).
    """
    # lista de arquivos spoofed (somente nomes)
    spoofed_files = {
        os.path.basename(p)
        for p in glob.glob(os.path.join(spoofed_dir, "statevectors_*.csv"))
    }

    Xs, ys = [], []
    for fp in glob.glob(os.path.join(test_dir, "statevectors_*.csv")):
        fname = os.path.basename(fp)
        label = 1 if fname in spoofed_files else 0

        df = pd.read_csv(fp)
        # mesma limpeza mínima do utils.load_tracks
        df = df.dropna(subset=['time','lat','lon','velocity','heading','vertrate','geoaltitude'])
        df['time'] = pd.to_datetime(df['time'].astype(int), unit='s', utc=True)
        if len(df) < 2:
            logger.debug("Pulando %s (menos de 2 pontos).", fname)
            continue

        # extrai e normaliza
        X = extract_features(df)
        X_scaled = scaler.transform(X)

        Xs.append(X_scaled)
        ys.extend([label] * X_scaled.shape[0])

    if not Xs:
        raise RuntimeError("Nenhum dado de teste válido foi carregado!")
    X_test = np.vstack(Xs)
    y_test = np.array(ys, dtype=int)
    logger.info("Dados de teste: %d amostras (%d spoofed, %d legit)",
                len(y_test), y_test.sum(), len(y_test)-y_test.sum())
    return X_test, y_test


def load_models(scaler_path, iforest_path, ocsvm_path, ae_model_path, ae_threshold_path):
    """Carrega scaler, IsolationForest, OC-SVM, Autoencoder e seu threshold."""
    scaler = joblib.load(scaler_path)
    logger.info("Scaler carregado de %s", scaler_path)

    iso = joblib.load(iforest_path)
    logger.info("IsolationForest carregado de %s", iforest_path)

    oc = joblib.load(ocsvm_path)
    logger.info("OneClassSVM carregado de %s", ocsvm_path)

    ae = load_model(ae_model_path)
    logger.info("Autoencoder carregado de %s", ae_model_path)

    with open(ae_threshold_path) as f:
        threshold = json.load(f)["threshold"]
    logger.info("Threshold do Autoencoder = %g", threshold)

    return scaler, iso, oc, ae, threshold


def get_scores_and_preds(iso, oc, ae, ae_threshold, X):
    """
    Retorna listas [scores], [preds] para cada modelo, com:
      - IsolationForest: alto score → mais anômalo
      - OC-SVM: idem
      - Autoencoder: MSE alto → anômalo
    """
    # IsolationForest
    score_iso = -iso.decision_function(X)
    pred_iso  = iso.predict(X) == -1

    # OC-SVM
    # note usamos score_samples para pontuar novos dados
    score_oc = -oc.score_samples(X)
    pred_oc  = oc.predict(X) == -1

    # Autoencoder
    recon = ae.predict(X)
    mse   = np.mean((recon - X)**2, axis=1)
    score_ae = mse
    pred_ae  = mse > ae_threshold

    return [score_iso, score_oc, score_ae], [pred_iso, pred_oc, pred_ae]


def plot_roc_pr(y_true, scores, labels, out_dir):
    """Plota e salva curvas ROC e Precision-Recall para cada modelo."""
    os.makedirs(out_dir, exist_ok=True)

    # ROC
    plt.figure(figsize=(6,6))
    for sc, lab in zip(scores, labels):
        fpr, tpr, _ = roc_curve(y_true, sc)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{lab} (AUC={roc_auc:.3f})")
    plt.plot([0,1],[0,1],'k--')
    plt.title("ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(out_dir, "roc_comparison.png")
    plt.savefig(roc_path)
    plt.close()
    logger.info("ROC salvo em %s", roc_path)

    # Precision-Recall
    plt.figure(figsize=(6,6))
    for sc, lab in zip(scores, labels):
        prec, rec, _ = precision_recall_curve(y_true, sc)
        pr_auc = auc(rec, prec)
        plt.plot(rec, prec, label=f"{lab} (AUPR={pr_auc:.3f})")
    plt.title("Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.tight_layout()
    pr_path = os.path.join(out_dir, "pr_comparison.png")
    plt.savefig(pr_path)
    plt.close()
    logger.info("PR salvo em %s", pr_path)


def main(args):
    # 1) Carrega modelos e scaler
    scaler, iso, oc, ae, ae_threshold = load_models(
        scaler_path    = args.scaler,
        iforest_path   = args.iforest,
        ocsvm_path     = args.ocsvm,
        ae_model_path  = args.autoencoder,
        ae_threshold_path = args.ae_threshold
    )

    # 2) Carrega dados de teste (com rótulos inferidos)
    X_test, y_test = load_test_data(
        test_dir    = args.test_dir,
        spoofed_dir = args.spoofed_dir,
        scaler      = scaler
    )

    # 3) Compute scores & preds
    scores, preds = get_scores_and_preds(iso, oc, ae, ae_threshold, X_test)

    # 4) Plot ROC e PR
    plot_roc_pr(y_test, scores, labels=args.labels, out_dir=args.output_dir)

    # 5) Classification reports
    logger.info("Classification Reports:")
    for pred, name in zip(preds, args.labels):
        report = classification_report(y_test, pred, digits=3)
        logger.info("--- %s ---\n%s", name, report)

    # 6) Salvar todas as predições em CSV
    summary_df = pd.DataFrame({
        "model": np.repeat(args.labels, len(y_test)),
        "true":  np.tile(y_test, len(args.labels)),
        "pred":   np.concatenate(preds)
    })
    summary_csv = os.path.join(args.output_dir, "all_preds.csv")
    summary_df.to_csv(summary_csv, index=False)
    logger.info("Predictions por ponto salvas em %s", summary_csv)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Avalia IsolationForest, OC-SVM e Autoencoder")
    p.add_argument("--test_dir",    required=True,
                   help="Pasta com todos os CSVs de teste (normais + spoofed)")
    p.add_argument("--spoofed_dir", required=True,
                   help="Pasta contendo apenas os CSVs spoofed para rotular")
    p.add_argument("--scaler",        required=True,
                   help="Caminho para scaler.pkl")
    p.add_argument("--iforest",       required=True,
                   help="Caminho para iso_forest.pkl")
    p.add_argument("--ocsvm",         required=True,
                   help="Caminho para oc_svm.pkl")
    p.add_argument("--autoencoder",   required=True,
                   help="Caminho para autoencoder.keras ou .h5")
    p.add_argument("--ae_threshold",  required=True,
                   help="Caminho para threshold.json do AE")
    p.add_argument("--output_dir",    default="evaluation_plots",
                   help="Diretório para salvar gráficos e relatórios")
    p.add_argument("--labels", nargs=3,
                   default=["IsolationForest","OC-SVM","Autoencoder"],
                   help="Rótulos para cada modelo na ordem")

    args = p.parse_args()
    main(args)
