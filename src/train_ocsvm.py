#!/usr/bin/env python3
# src/train_ocsvm.py

import os
import joblib
import numpy as np
from sklearn.svm import OneClassSVM

def load_data(npz_path):
    """Carrega X e rotas de um arquivo .npz."""
    data = np.load(npz_path, allow_pickle=True)
    X = data['X']
    routes = data.get('routes', None)
    return X, routes

def train_oneclass_svm(train_npz, val_npz, out_dir,
                       kernel='rbf', nu=0.01, gamma='auto'):
    """
    Treina um One-Class SVM em X_train, avalia em X_val e salva o modelo.
    Parâmetros:
      - kernel: 'rbf', 'linear', etc.
      - nu: fração máxima de outliers (0 < nu < 1)
      - gamma: parâmetro de kernel ('scale', 'auto' ou float)
    """
    # 1) carregar dados
    X_train, _ = load_data(train_npz)
    X_val, _   = load_data(val_npz)

    # 2) instanciar e treinar
    oc = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
    print(f"🟢 Treinando One-Class SVM (kernel={kernel}, nu={nu}, gamma={gamma})...")
    oc.fit(X_train)

    # 3) criar saída
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, 'oc_svm.pkl')
    joblib.dump(oc, model_path)
    print(f"✔️ Modelo salvo em {model_path}")

    # 4) avaliação no conjunto de validação
    preds = oc.predict(X_val)   # +1 normal, -1 anomalia
    n_abnormal = np.sum(preds == -1)
    print(f"📊 Validação: {n_abnormal} de {len(X_val)} pontos marcados como anomalia.")

    # 5) salvar preds para análise posterior
    np.savez(os.path.join(out_dir, 'ocsvm_val_preds.npz'), preds=preds)
    print(f"✔️ Predições de validação salvas em {out_dir}/ocsvm_val_preds.npz")

    return oc

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Treina One-Class SVM para detecção de spoofing."
    )
    parser.add_argument(
        '--train',
        default='data/raw/tracks/train.npz',
        help='Caminho para train.npz (X já normalizado)'
    )
    parser.add_argument(
        '--val',
        default='data/raw/tracks/val.npz',
        help='Caminho para val.npz (X já normalizado)'
    )
    parser.add_argument(
        '--out_dir',
        default='models/ocsvm',
        help='Pasta para salvar oc_svm.pkl e preds de validação'
    )
    parser.add_argument(
        '--kernel',
        default='rbf',
        choices=['linear','poly','rbf','sigmoid'],
        help='Kernel do SVM (padrão: rbf)'
    )
    parser.add_argument(
        '--nu',
        type=float,
        default=0.01,
        help='Proporção estimada de outliers (0 < nu < 1, padrão: 0.01)'
    )
    parser.add_argument(
        '--gamma',
        default='auto',
        help="Parâmetro gamma para kernels ('scale','auto' ou float, padrão: 'auto')"
    )
    args = parser.parse_args()

    train_oneclass_svm(
        train_npz=args.train,
        val_npz=args.val,
        out_dir=args.out_dir,
        kernel=args.kernel,
        nu=args.nu,
        gamma=args.gamma
    )
