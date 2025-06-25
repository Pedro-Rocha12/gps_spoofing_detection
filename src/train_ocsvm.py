#!/usr/bin/env python3
# src/train_ocsvm.py

import os
import logging
import joblib
import numpy as np
import argparse
from sklearn.svm import OneClassSVM

# --- logging setup ---
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)

def load_data(npz_path):
    """Carrega X (normalizado) e rotas de um arquivo .npz."""
    data = np.load(npz_path, allow_pickle=True)
    X = data['X']
    routes = data.get('routes', None)
    logger.info("  → Carregado %s: %d amostras × %d features", npz_path, *X.shape)
    return X, routes

def train_oneclass_svm(train_npz, val_npz, test_npz, out_dir,
                       kernel='rbf', nu=0.01, gamma='auto'):
    """
    Treina um One-Class SVM em X_train, avalia em X_val (e opcionalmente em X_test)
    e salva modelo, preds e scores.
    """
    # 1) Carregar dados
    X_train, _ = load_data(train_npz)
    X_val,   _ = load_data(val_npz)
    if test_npz:
        X_test, _ = load_data(test_npz)
    else:
        X_test = None

    # 2) Instanciar e treinar
    logger.info("Instanciando OneClassSVM (kernel=%s, nu=%.4f, gamma=%s)", kernel, nu, gamma)
    oc = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma, verbose=False)
    logger.info("Treinando OneClassSVM em %d amostras...", X_train.shape[0])
    oc.fit(X_train)

    # 3) Criar saída
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, 'oc_svm.pkl')
    joblib.dump(oc, model_path)
    logger.info("✔️ Modelo salvo em %s", model_path)

    # 4) Avaliação no conjunto de validação
    logger.info("Avaliação em validação...")
    preds_val = oc.predict(X_val)            # +1 normal, -1 anomalia
    scores_val = oc.score_samples(X_val)     # quanto maior, mais normal
    n_abnormal = np.sum(preds_val == -1)
    logger.info("📊 Validação: %d de %d pontos anômalos (%.2f%%)",
                n_abnormal, len(preds_val), 100 * n_abnormal / len(preds_val))

    np.savez(os.path.join(out_dir, 'ocsvm_val.npz'),
             preds=preds_val, scores=scores_val)
    logger.info("✔️ preds e scores de validação salvos em %s", out_dir)

    # 5) Avaliação no conjunto de teste (se fornecido)
    if X_test is not None:
        logger.info("Avaliação em teste...")
        preds_test = oc.predict(X_test)
        scores_test = oc.score_samples(X_test)
        n_abnormal_t = np.sum(preds_test == -1)
        logger.info("📊 Teste:    %d de %d pontos anômalos (%.2f%%)",
                    n_abnormal_t, len(preds_test), 100 * n_abnormal_t / len(preds_test))
        np.savez(os.path.join(out_dir, 'ocsvm_test.npz'),
                 preds=preds_test, scores=scores_test)
        logger.info("✔️ preds e scores de teste salvos em %s", out_dir)

    return oc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Treina One-Class SVM para detecção de spoofing em dados ADS-B."
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
        '--test',
        default=None,
        help='(opcional) caminho para test.npz'
    )
    parser.add_argument(
        '--out_dir',
        default='models/ocsvm',
        help='Pasta para salvar oc_svm.pkl, preds e scores'
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
        help='Fração estimada de outliers (0 < nu < 1, padrão: 0.01)'
    )
    parser.add_argument(
        '--gamma',
        default='auto',
        help="Parâmetro gamma para kernel ('scale','auto' ou float, padrão: 'auto')"
    )

    args = parser.parse_args()

    train_oneclass_svm(
        train_npz=args.train,
        val_npz=args.val,
        test_npz=args.test,
        out_dir=args.out_dir,
        kernel=args.kernel,
        nu=args.nu,
        gamma=args.gamma
    )
