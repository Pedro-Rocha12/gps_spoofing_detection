#!/usr/bin/env python3
# src/analyze_iforest.py

import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def load_npz(path):
    """Carrega X e, opcionalmente, routes de um .npz"""
    data = np.load(path, allow_pickle=True)
    X = data['X']
    routes = data['routes'] if 'routes' in data else None
    return X, routes

def plot_roc_pr(y_true, scores, label, out_dir):
    """Plota e salva ROC e PR curves para um único modelo."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true, scores)
    pr_auc = auc(rec, prec)

    # ROC
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {label}")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    fname = os.path.join(out_dir, f"roc_{label}.png")
    plt.savefig(fname)
    plt.close()

    # Precision-Recall
    plt.figure(figsize=(6,6))
    plt.plot(rec, prec, label=f"{label} (AUPR={pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall — {label}")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    fname = os.path.join(out_dir, f"pr_{label}.png")
    plt.savefig(fname)
    plt.close()

def evaluate_saved_scores(val_scores_path, val_preds_path=None, y_true=None, label="IForest", out_dir="analysis_iforest"):
    """
    Carrega scores salvos (.npz) e, se y_true informado, plota ROC/PR.
    val_scores_path: .npz com array 'scores'
    y_true: array binário de ground truth (1=anomalia, 0=normal)
    """
    os.makedirs(out_dir, exist_ok=True)
    data = np.load(val_scores_path)
    scores = data['scores']
    if y_true is None:
        print("Sem ground‐truth fornecida, abortando ROC/PR.")
        return
    plot_roc_pr(y_true, scores, label, out_dir)
    print(f"✅ ROC/PR para {label} salvos em {out_dir}/")

def tune_contamination(X_train, X_val, y_val,
                       contamination_list,
                       n_estimators=100,
                       random_state=42,
                       out_dir="analysis_iforest"):
    """
    Varre diferentes valores de contamination e calcula AUC no validation set.
    Plota AUC vs contamination.
    """
    os.makedirs(out_dir, exist_ok=True)
    aucs = []
    for c in contamination_list:
        iso = IsolationForest(
            contamination=c,
            n_estimators=n_estimators,
            random_state=random_state,
            verbose=0
        )
        iso.fit(X_train)
        scores = -iso.decision_function(X_val)  # usamos -decision_function para que altas scores = mais anômalo
        fpr, tpr, _ = roc_curve(y_val, scores)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        print(f"contamination={c:.4f} -> AUC={roc_auc:.4f}")

    # Plot
    plt.figure(figsize=(8,4))
    plt.plot(contamination_list, aucs, marker='o')
    plt.xlabel("contamination")
    plt.ylabel("ROC AUC")
    plt.title("Tuning de contamination (IsolationForest)")
    plt.grid(True)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "tuning_contamination.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"✅ Gráfico de tuning salvo em {fig_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Análise e tuning do IsolationForest"
    )
    parser.add_argument("--train", default="data/raw/tracks/train.npz",
                        help="Caminho para train.npz")
    parser.add_argument("--val", default="data/raw/tracks/val.npz",
                        help="Caminho para val.npz")
    parser.add_argument("--test_scores", default="models/iforest/iforest_teste_scores.npz",
                        help=".npz com scores do teste")
    parser.add_argument("--val_scores", default="models/iforest/iforest_validação_scores.npz",
                        help=".npz com scores da validação")
    parser.add_argument("--y_val", default=None,
                        help=".npz com ground-truth y_val (array 'y') para plotar ROC/PR")
    parser.add_argument("--out_dir", default="analysis_iforest",
                        help="Pasta de saída para gráficos")
    parser.add_argument("--cont_list", nargs='+', type=float,
                        default=[0.005,0.01,0.02,0.05,0.1],
                        help="Lista de contamination para tuning")
    args = parser.parse_args()

    # 1) Carregar dados
    X_train, _ = load_npz(args.train)
    X_val, routes_val = load_npz(args.val)

    # Se y_val for passado como .npz
    y_val = None
    if args.y_val:
        data = np.load(args.y_val, allow_pickle=True)
        y_val = data['y']

    # 2) Tuning de contamination
    tune_contamination(
        X_train, X_val, y_val,
        contamination_list=args.cont_list,
        n_estimators=100,
        random_state=42,
        out_dir=args.out_dir
    )

    # 3) Plot ROC/PR usando scores já salvos
    if y_val is not None:
        evaluate_saved_scores(
            args.val_scores,
            y_true=y_val,
            label="IsolationForest Val",
            out_dir=args.out_dir
        )
        # se quiser plotar teste, carregue y_test e scores de teste
        # evaluate_saved_scores(
        #     args.test_scores,
        #     y_true=y_test,
        #     label="IsolationForest Test",
        #     out_dir=args.out_dir
        # )

if __name__ == "__main__":
    main()
