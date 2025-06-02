# src/models.py
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, Model

def train_isolation_forest(X: np.ndarray):
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    iso = IsolationForest(contamination=0.01, random_state=42).fit(Xs)
    return scaler, iso

def get_lof_model(X, n_neighbors=20):
    """
    Recebe X (DataFrame ou array), treina um StandardScaler + LOF(novelty=True)
    Retorna (scaler, fitted_lof)
    """
    # 1) escala
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # 2) LOF em modo novelty para poder usar decision_function()
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    lof.fit(Xs)

    return scaler, lof

def train_autoencoder(X: np.ndarray, encoding_dim=8, epochs=10):
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    inp = layers.Input(shape=(Xs.shape[1],))
    encoded = layers.Dense(encoding_dim, activation='relu')(inp)
    decoded = layers.Dense(Xs.shape[1], activation='linear')(encoded)
    ae = Model(inp, decoded)
    ae.compile(optimizer='adam', loss='mse')
    ae.fit(Xs, Xs, epochs=epochs, batch_size=256, verbose=1)
    return scaler, ae

def score_iso(scaler, iso, X: np.ndarray):
    Xs = scaler.transform(X)
    return iso.decision_function(Xs)

def score_lof(scaler, lof: LocalOutlierFactor, X):
    Xs = scaler.transform(X)
    return lof.decision_function(Xs)

def score_ae(scaler, ae, X: np.ndarray):
    Xs = scaler.transform(X)
    recons = ae.predict(Xs)
    return np.mean((Xs - recons)**2, axis=1)
