#!/usr/bin/env python3
# src/train_iforest.py

import os
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

def load_data(npz_path):
    """Carrega X e rotas de um arquivo .npz."""
    data = np.load(npz_path, allow_pickle=True)
    X = data['X']
    routes = data['routes']
    return X, routes

def train_isolation_forest(train_npz, val_npz, out_dir,
                           contamination=0.01, n_estimators=100,
                           random_state=42):
    """
    Treina um IsolationForest em X_train, avalia em X_val e salva o modelo.
    """
    # 1) carregar dados
    X_train, _ = load_data(train_npz)
    X_val, _   = load_data(val_npz)

    # 2) instanciar e treinar
    iso = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
        verbose=1
    )
    print("🟢 Treinando IsolationForest...")
    iso.fit(X_train)

    # 3) salvar o modelo
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, 'iso_forest.pkl')
    joblib.dump(iso, model_path)
    print(f"✔️ Modelo salvo em {model_path}")

    # 4) avaliação simples no conjunto de validação
    scores = iso.decision_function(X_val)
    preds  = iso.predict(X_val)  # +1 normal, -1 anomalia
    n_abnormal = np.sum(preds == -1)
    print(f"📊 Validação: {n_abnormal} de {len(X_val)} pontos marcados como anomalia.")

    # opcional: salvar scores e preds
    np.savez(os.path.join(out_dir, 'iforest_val_scores.npz'),
             scores=scores, preds=preds)
    print(f"✔️ Scores de validação salvos em {out_dir}/iforest_val_scores.npz")

    return iso

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Treina IsolationForest para detecção de spoofing."
    )
    parser.add_argument(
        '--train',
        default='data/raw/tracks/train.npz',
        help='Caminho para train.npz'
    )
    parser.add_argument(
        '--val',
        default='data/raw/tracks/val.npz',
        help='Caminho para val.npz'
    )
    parser.add_argument(
        '--out_dir',
        default='models',
        help='Pasta para salvar iso_forest.pkl e arquivos de validação'
    )
    parser.add_argument(
        '--contamination',
        type=float,
        default=0.01,
        help='Proporção estimada de anomalias (padrão: 0.01)'
    )
    parser.add_argument(
        '--n_estimators',
        type=int,
        default=100,
        help='Número de árvores (padrão: 100)'
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Seed para reprodutibilidade'
    )
    args = parser.parse_args()

    train_isolation_forest(
        train_npz=args.train,
        val_npz=args.val,
        out_dir=args.out_dir,
        contamination=args.contamination,
        n_estimators=args.n_estimators,
        random_state=args.random_state
    )
