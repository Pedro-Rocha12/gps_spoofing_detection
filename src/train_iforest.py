#!/usr/bin/env python3
# src/train_iforest.py

import os
import joblib
import numpy as np
import logging
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_data(npz_path):
    """Carrega X e rotas de um arquivo .npz."""
    data = np.load(npz_path, allow_pickle=True)
    return data['X'], data.get('routes', None)

def train_isolation_forest(
    train_npz: str,
    val_npz: str,
    test_npz: str,
    out_dir: str,
    contamination: float = 0.01,
    n_estimators: int = 100,
    random_state: int = 42
):
    """
    Treina um IsolationForest em X_train, avalia em X_val e X_test e salva o modelo.
    """
    # 1) carrega dados
    X_train, _ = load_data(train_npz)
    X_val,   _ = load_data(val_npz)
    X_test,  _ = load_data(test_npz)

    # 2) instancia e treina
    iso = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
        verbose=1
    )
    logger.info(
        "Treinando IsolationForest (contamination=%.4f, n_estimators=%d, random_state=%d)...",
        contamination, n_estimators, random_state
    )
    iso.fit(X_train)

    # 3) salva o modelo
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, 'iso_forest.pkl')
    joblib.dump(iso, model_path)
    logger.info("✔️ Modelo salvo em %s", model_path)

    # 4) função auxiliar de avaliação
    def evaluate_split(X: np.ndarray, split_name: str):
        scores = iso.decision_function(X)
        preds  = iso.predict(X)  # +1 = normal, -1 = anomalia
        n_abnormal = int((preds == -1).sum())
        logger.info("🔍 %s: %d/%d pontos anômalos", split_name, n_abnormal, len(X))
        scores_path = os.path.join(out_dir, f'iforest_{split_name.lower()}_scores.npz')
        np.savez(scores_path, scores=scores, preds=preds)
        logger.info("✔️ Scores %s salvos em %s", split_name, scores_path)

    # 5) avalia em validação e teste
    evaluate_split(X_val,  "Validação")
    evaluate_split(X_test, "Teste")

    return iso

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Treina IsolationForest para detecção de spoofing."
    )
    parser.add_argument(
        '--train', default='data/raw/tracks/train.npz',
        help='Caminho para train.npz'
    )
    parser.add_argument(
        '--val', default='data/raw/tracks/val.npz',
        help='Caminho para val.npz'
    )
    parser.add_argument(
        '--test', default='data/raw/tracks/test.npz',
        help='Caminho para test.npz'
    )
    parser.add_argument(
        '--out_dir', default='models/iforest',
        help='Pasta para salvar iso_forest.pkl e scores'
    )
    parser.add_argument(
        '--contamination', type=float, default=0.01,
        help='Proporção estimada de anomalias (padrão: 0.01)'
    )
    parser.add_argument(
        '--n_estimators', type=int, default=100,
        help='Número de árvores (padrão: 100)'
    )
    parser.add_argument(
        '--random_state', type=int, default=42,
        help='Seed para reprodutibilidade'
    )

    args = parser.parse_args()

    train_isolation_forest(
        train_npz=args.train,
        val_npz=args.val,
        test_npz=args.test,
        out_dir=args.out_dir,
        contamination=args.contamination,
        n_estimators=args.n_estimators,
        random_state=args.random_state
    )
