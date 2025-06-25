#!/usr/bin/env python3
# src/train_autoencoder.py

import os
import json
import logging

import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    inp = layers.Input(shape=(input_dim,), name="ae_input")
    h = layers.Dense(16, activation='relu', name="encoder_h1")(inp)
    code = layers.Dense(8, activation='relu', name="bottleneck")(h)
    h2 = layers.Dense(16, activation='relu', name="decoder_h1")(code)
    out = layers.Dense(input_dim, activation=None, name="ae_output")(h2)
    auto = models.Model(inputs=inp, outputs=out, name='autoencoder')
    auto.compile(optimizer='adam', loss='mse')
    return auto

def train_autoencoder(
    train_npz: str,
    val_npz: str,
    out_dir: str,
    epochs: int = 50,
    batch_size: int = 256,
    threshold_percentile: float = 99.0,
    patience: int = 5
):
    # 1) Carrega dados
    X_train, _ = load_data(train_npz)
    X_val,   _ = load_data(val_npz)
    logger.info("Dados carregados: train=%s (%d×%d), val=%s (%d×%d)",
                train_npz, *X_train.shape,
                val_npz,   *X_val.shape)

    # 2) Constrói o modelo
    input_dim = X_train.shape[1]
    auto = build_autoencoder(input_dim)
    auto.summary()

    # 3) Callbacks: salvar melhor e early stopping
    os.makedirs(out_dir, exist_ok=True)
    best_model_path = os.path.join(out_dir, 'best_autoencoder.keras')
    mc = callbacks.ModelCheckpoint(
        best_model_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    es = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    # 4) Treina
    logger.info("Iniciando treino por até %d épocas (batch=%d)…", epochs, batch_size)
    history = auto.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, X_val),
        callbacks=[mc, es],
        verbose=2
    )

    # 5) Plota curva de perda
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(loss) + 1)
    plt.figure(figsize=(8,4))
    plt.plot(epochs_range, loss,    label='train loss')
    plt.plot(epochs_range, val_loss,label='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.tight_layout()
    loss_plot_path = os.path.join(out_dir, 'loss_curve.png')
    plt.savefig(loss_plot_path)
    plt.close()
    logger.info("📈 Curva de perda salva em %s", loss_plot_path)

    # 6) Calcula MSE de validação para definir threshold
    recon_val = auto.predict(X_val)
    mse = np.mean(np.square(recon_val - X_val), axis=1)
    threshold = float(np.percentile(mse, threshold_percentile))
    logger.info("ℹ️ Threshold (percentil %.1f): %.6f", threshold_percentile, threshold)

    # 7) Salva artefatos finais
    model_path = os.path.join(out_dir, 'autoencoder.keras')
    auto.save(model_path, save_format='keras')
    with open(os.path.join(out_dir, 'threshold.json'), 'w') as f:
        json.dump({'threshold': threshold}, f)
    np.savez(os.path.join(out_dir, 'val_mse.npz'), mse=mse)

    hist_path = os.path.join(out_dir, 'history.json')
    with open(hist_path, 'w') as f:
        json.dump(history.history, f)

    logger.info("✔️ Modelo salvo em       %s", model_path)
    logger.info("✔️ Threshold salvo em    %s", os.path.join(out_dir, 'threshold.json'))
    logger.info("✔️ MSE de validação em   %s", os.path.join(out_dir, 'val_mse.npz'))
    logger.info("✔️ Histórico salvo em    %s", hist_path)

if __name__ == '__main__':
    import argparse

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
        help='Pasta de saída para artefatos'
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
        help='Tamanho do batch (padrão: 256)'
    )
    parser.add_argument(
        '--threshold_percentile',
        type=float,
        default=99.0,
        help='Percentil para definição de limiar (padrão: 99)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=5,
        help='Paciencia do early stopping (padrão: 5)'
    )
    args = parser.parse_args()

    train_autoencoder(
        train_npz=args.train,
        val_npz=args.val,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        threshold_percentile=args.threshold_percentile,
        patience=args.patience
    )
