#!/usr/bin/env python3
# src/prepare_val_labels.py

import os
import numpy as np
import joblib
import argparse

def main(val_npz, filtered_dir, excluded_dir, out_path):
    # 1) carrega X e routes do val.npz
    data = np.load(val_npz, allow_pickle=True)
    X_val = data['X']
    routes = data['routes']  # array de strings: nome de cada rota

    # 2) monta y_val: 0 para qualquer rota em filtered_dir, 1 em excluded_dir
    y = []
    filtered = set(os.listdir(filtered_dir))
    excluded = set(os.listdir(excluded_dir))
    for r in routes:
        if r in excluded:
            y.append(1)
        elif r in filtered:
            y.append(0)
        else:
            # se não encontrou em nenhum, trate como legítimo (ou lance warning)
            y.append(0)
    y = np.array(y, dtype=int)

    # 3) salva em npz
    np.savez(out_path, y=y)
    print(f"✔️ y_val salvo em {out_path} — {y.sum()} spoofed, {len(y)-y.sum()} legit")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--val_npz',       default='data/raw/tracks/val.npz')
    p.add_argument('--filtered_dir',  default='data/tracks_filtradas')
    p.add_argument('--excluded_dir',  default='data/tracks')
    p.add_argument('--out',           default='data/split_output/val_labels.npz')
    args = p.parse_args()
    main(args.val_npz, args.filtered_dir, args.excluded_dir, args.out)
