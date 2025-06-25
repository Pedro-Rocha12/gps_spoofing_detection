#!/usr/bin/env python3
# evaluation_routes.py

import os
import argparse
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)

def load_split_routes(split_npz: str):
    """
    Carrega X e routes de split_output/test.npz.
    Retorna (routes: List[str], X: np.ndarray).
    """
    data = np.load(split_npz, allow_pickle=True)
    routes = data['routes']
    X = data['X']
    return routes, X

def load_models(scaler_path, iso_path, oc_path, ae_model_path, ae_threshold_path):
    scaler = joblib.load(scaler_path)
    iso    = joblib.load(iso_path)
    oc     = joblib.load(oc_path)
    ae     = load_model(ae_model_path)
    with open(ae_threshold_path) as f:
        ae_threshold = json.load(f)['threshold']
    return scaler, iso, oc, ae, ae_threshold

def compute_pointwise(routes, X, iso, oc, ae, ae_threshold):
    """
    Para cada ponto em X, calcula score e pred para cada modelo.
    Returns dicts preds and scores keyed pelos nomes dos modelos.
    """
    # IsolationForest
    sc_iso = -iso.decision_function(X)
    pd_iso = (iso.predict(X) == -1).astype(int)
    # One-Class SVM
    sc_oc  = -oc.score_samples(X)
    pd_oc  = (oc.predict(X) == -1).astype(int)
    # Autoencoder
    recon = ae.predict(X, verbose=0)
    mse   = np.mean((recon - X)**2, axis=1)
    sc_ae = mse
    pd_ae = (mse > ae_threshold).astype(int)

    return {
        'IsolationForest': (pd_iso, sc_iso),
        'OC-SVM':          (pd_oc,  sc_oc),
        'Autoencoder':     (pd_ae,  sc_ae)
    }

def build_route_df(routes, model_preds, spoofed_dir):
    """
    Agrupa por rota e calcula:
      - count: total de pontos
      - n_anom_model, frac_anom_model para cada modelo
      - true_label: 1 se arquivo em spoofed_dir, 0 caso contrário
    """
    df = pd.DataFrame({'route': routes})
    df['point_idx'] = df.index

    grp = df.groupby('route')['point_idx'].count().rename('count').to_frame()
    for name, (preds, _) in model_preds.items():
        df['pred_'+name] = preds
        agg = df.groupby('route')['pred_'+name].agg(['sum'])
        grp['n_anom_'+name]  = agg['sum']
        grp['frac_anom_'+name] = agg['sum'] / grp['count']

    # true label de rota
    spoofed = set(os.listdir(spoofed_dir))
    grp['true_label'] = grp.index.map(lambda fn: 1 if fn in spoofed else 0)

    return grp.reset_index()

def plot_per_model_routes(df, model, threshold, out_dir):
    """
    Para um modelo, gera e salva:
      - classification_report por rota
      - histograma de frac_anom
      - ROC por rota
      - Precision-Recall por rota
    """
    os.makedirs(out_dir, exist_ok=True)
    frac = df['frac_anom_'+model]
    y_true = df['true_label']
    y_pred = (frac >= threshold).astype(int)

    # classification report
    print(f"\n--- Report por rota ({model}, thresh={threshold:.2%}) ---")
    print(classification_report(y_true, y_pred, digits=3))

    # histograma
    plt.figure(figsize=(6,4))
    plt.hist([frac[y_true==0], frac[y_true==1]],
             bins=20, stacked=True, label=['normal','spoofed'])
    plt.axvline(threshold, color='k', linestyle='--', label=f'th={threshold:.2%}')
    plt.title(f'Fração de pontos anômalos por rota — {model}')
    plt.xlabel('Fração de pontos anômalos')
    plt.ylabel('Número de rotas')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'hist_routes_{model}.png'))
    plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(y_true, frac)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr, label=f'AUC={roc_auc:.3f}')
    plt.plot([0,1],[0,1],'k--')
    plt.title(f'ROC por rota — {model}')
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'roc_routes_{model}.png'))
    plt.close()

    # Precision-Recall
    prec, rec, _ = precision_recall_curve(y_true, frac)
    pr_auc = auc(rec, prec)
    plt.figure(figsize=(5,5))
    plt.plot(rec, prec, label=f'AUPR={pr_auc:.3f}')
    plt.title(f'Precision-Recall por rota — {model}')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'pr_routes_{model}.png'))
    plt.close()

def plot_comparison(df, models, out_dir):
    """
    Plota curvas ROC e PR comparando todos os modelos no nível de rota.
    Usa frac_anom_model como score.
    """
    os.makedirs(out_dir, exist_ok=True)
    y = df['true_label']
    # ROC compara
    plt.figure(figsize=(5,5))
    for m in models:
        frac = df['frac_anom_'+m]
        fpr, tpr, _ = roc_curve(y, frac)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{m} (AUC={roc_auc:.3f})")
    plt.plot([0,1],[0,1],'k--')
    plt.title("ROC Comparison (per-route)")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_routes_comparison.png"))
    plt.close()

    # PR compara
    plt.figure(figsize=(5,5))
    for m in models:
        frac = df['frac_anom_'+m]
        prec, rec, _ = precision_recall_curve(y, frac)
        pr_auc = auc(rec, prec)
        plt.plot(rec, prec, label=f"{m} (AUPR={pr_auc:.3f})")
    plt.title("Precision-Recall Comparison (per-route)")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pr_routes_comparison.png"))
    plt.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Avalia os modelos no nível de rota (route‐level)"
    )
    p.add_argument(
        "--split",      required=True,
        help="split_output/test.npz (com arrays X e routes)"
    )
    p.add_argument(
        "--spoofed_dir", required=True,
        help="pasta spoofed_tracks/ com arquivos rotas spoofed"
    )
    p.add_argument(
        "--scaler",     required=True,
        help="scaler.pkl usado para normalizar X"
    )
    p.add_argument(
        "--iforest",    required=True,
        help="iso_forest.pkl"
    )
    p.add_argument(
        "--ocsvm",      required=True,
        help="oc_svm.pkl"
    )
    p.add_argument(
        "--autoencoder",      required=True,
        help="autoencoder.keras"
    )
    p.add_argument(
        "--ae_threshold",required=True,
        help="threshold.json do Autoencoder"
    )
    p.add_argument(
        "--threshold",  type=float, default=0.05,
        help="limiar de fração de pontos anômalos para classificar rota (default=0.05)"
    )
    p.add_argument(
        "--out_dir", default="route_evaluation",
        help="diretório para salvar outputs"
    )
    args = p.parse_args()

    # 1) load split
    routes, X = load_split_routes(args.split)
    # 2) load models
    scaler, iso, oc, ae, ae_threshold = load_models(
        args.scaler,
        args.iforest,
        args.ocsvm,
        args.autoencoder,
        args.ae_threshold
    )
    # 3) (re)normalize X se necessário
    X = scaler.transform(X)
    # 4) compute preds & scores ponto‐a‐ponto
    model_preds = compute_pointwise(routes, X, iso, oc, ae, ae_threshold)
    # 5) agrupar por rota
    df_routes = build_route_df(routes, model_preds, args.spoofed_dir)
    # 6) avaliar cada modelo separadamente
    for model in ["IsolationForest","OC-SVM","Autoencoder"]:
        plot_per_model_routes(df_routes, model, args.threshold, args.out_dir)
    # 7) plotar comparação entre modelos
    plot_comparison(df_routes,
                    ["IsolationForest","OC-SVM","Autoencoder"],
                    args.out_dir)
    # 8) salvar tabela por rota
    df_routes.to_csv(os.path.join(args.out_dir, "route_level_summary.csv"), index=False)
    print(f"\n✅ Route‐level summary salvo em {args.out_dir}/route_level_summary.csv")
