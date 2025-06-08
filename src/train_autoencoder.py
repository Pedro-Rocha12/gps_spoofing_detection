#!/usr/bin/env python3
# src/train_autoencoder.py

import os
import json
import joblib
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
import argparse

def load_data(npz_path):
    """Carrega X e rotas de um arquivo .npz."""
    data = np.load(npz_path, allow_pickle=True)
    return data['X'], data.get('routes', None)

def build_autoencoder(input_dim):
    """
    Retorna um modelo autoencoder simples:
    encoder: Dense(16) → Dense(8)
    decoder: Dense(16) → Dense(input_dim)
    """
    inp = layers.Input(shape=(input_dim,))
    h = layers.Dense(16, activation='relu')(inp)
    code = layers.Dense(8, activation='relu')(h)
    h2 = layers.Dense(16, activation='relu')(code)
    out = layers.Dense(input_dim, activation=None)(h2)
    auto = models.Model(inputs=inp, outputs=out, name='autoencoder')
    auto.compile(optimizer='adam', loss='mse')
    return auto

def train_autoencoder(train_npz, val_npz, out_dir,
                      epochs=50, batch_size=256, threshold_percentile=99):
    # 1) Carrega dados
    X_train, _ = load_data(train_npz)
    X_val, _   = load_data(val_npz)
    input_dim = X_train.shape[1]

    # 2) Constrói o modelo
    auto = build_autoencoder(input_dim)
    auto.summary()

    # 3) Treina
    history = auto.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, X_val),
        verbose=2
    )

    # 4) Calcula MSE na validação
    recon_val = auto.predict(X_val)
    mse = np.mean(np.square(recon_val - X_val), axis=1)

    # 5) Define limiar
    threshold = np.percentile(mse, threshold_percentile)
    print(f"ℹ️  Threshold (percentil {threshold_percentile}): {threshold:.6f}")

    # 6) Salva artefatos
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, 'autoencoder.h5')
    th_path    = os.path.join(out_dir, 'threshold.json')
    mse_path   = os.path.join(out_dir, 'val_mse.npz')

    auto.save(model_path)
    with open(th_path, 'w') as f:
        json.dump({'threshold': float(threshold)}, f)
    np.savez(mse_path, mse=mse)

    print(f"✔️ Modelo salvo em       {model_path}")
    print(f"✔️ Threshold salvo em    {th_path}")
    print(f"✔️ MSE de validação em   {mse_path}")

    return auto, history

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Treina Autoencoder para detecção de spoofing"
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
        default='models/autoencoder',
        help='Pasta de saída para salvar autoencoder.h5, threshold.json e val_mse.npz'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Número de épocas (padrão: 50)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Tamanho de batch (padrão: 256)'
    )
    parser.add_argument(
        '--threshold_percentile',
        type=float,
        default=99.0,
        help='Percentil para definição de limiar (padrão: 99)'
    )
    args = parser.parse_args()

    train_autoencoder(
        train_npz=args.train,
        val_npz=args.val,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        threshold_percentile=args.threshold_percentile
    )
