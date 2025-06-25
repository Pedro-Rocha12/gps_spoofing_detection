#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils import load_tracks, extract_features, FEATURE_NAMES

def prepare_npz(splits_dir: str, out_dir: str):
    """
    A partir de um diretório contendo subpastas train/, val/ e test/ com CSVs de statevectors:
      1) Carrega e extrai features de cada rota em cada split;
      2) Concatena tudo em X_train, X_val, X_test e rotas associadas;
      3) Ajusta um StandardScaler em X_train e aplica em todos os splits;
      4) Salva train.npz, val.npz, test.npz, scaler.pkl e feature_names.json em out_dir.
    """
    data = {}
    for split in ("train", "val", "test"):
        folder = os.path.join(splits_dir, split)
        if not os.path.isdir(folder):
            raise ValueError(f"Diretório não encontrado: {folder}")
        names, dfs = load_tracks(folder)
        X_list, routes = [], []
        for name, df in zip(names, dfs):
            X = extract_features(df)
            if X.size:
                X_list.append(X)
                routes.extend([name] * X.shape[0])
        if X_list:
            X_all = np.vstack(X_list)
            routes_all = np.array(routes, dtype=object)
        else:
            X_all = np.empty((0, len(FEATURE_NAMES)))
            routes_all = np.empty((0,), dtype=object)
        data[split] = {"X": X_all, "routes": routes_all}

    # 1) Fit scaler no treino
    scaler = StandardScaler().fit(data["train"]["X"])
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    # salva nomes de feature
    with open(os.path.join(out_dir, "feature_names.json"), "w") as f:
        json.dump(FEATURE_NAMES, f)

    # 2) Transforma e salva cada split
    for split in ("train", "val", "test"):
        X = data[split]["X"]
        routes = data[split]["routes"]
        unique_routes = np.unique(routes).size
        X_scaled = scaler.transform(X)

        npz_path = os.path.join(out_dir, f"{split}.npz")
        np.savez(
            npz_path,
            X=X_scaled,
            routes=routes
        )

        print(f"\n→ Split: {split}")
        print(f"   • Amostras (pontos): {X.shape[0]}")
        print(f"   • Rotas únicas:      {unique_routes}")
        print(f"✔️  {split}.npz salvo em {npz_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Prepara .npz de train/val/test a partir de pastas com CSVs de statevectors"
    )
    parser.add_argument(
        "splits_dir",
        help="Diretório contendo subpastas train/, val/, test/ com CSVs"
    )
    parser.add_argument(
        "out_dir",
        help="Pasta de saída para .npz, scaler.pkl e feature_names.json"
    )
    args = parser.parse_args()
    prepare_npz(args.splits_dir, args.out_dir)
