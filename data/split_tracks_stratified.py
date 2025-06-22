#!/usr/bin/env python3
# split_tracks_stratified.py
# Divide estratificadamente as rotas em treino/val/test e move os respectivos CSVs.

import os
import glob
import shutil
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def stratified_split_and_move(
    tracks_dir: str,
    summary_csv: str,
    output_dir: str,
    stratify_col: str,
    n_bins: int = 5,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42
):
    # 1) lê o CSV de métricas
    df = pd.read_csv(summary_csv)
    if 'route' not in df.columns or stratify_col not in df.columns:
        raise ValueError(f"O CSV deve conter colunas 'route' e '{stratify_col}'")
    
    # 2) cria estratos com qcut
    df['strata'] = pd.qcut(df[stratify_col], q=n_bins, duplicates='drop')
    
    # 3) split em teste (10%) e resto (90%)
    df_rest, df_test = train_test_split(
        df,
        test_size=test_size,
        stratify=df['strata'],
        random_state=random_state
    )
    # 4) split do resto em treino (80/90≈88.9%) e val (10/90≈11.1%)
    val_rel = val_size / (train_size + val_size)
    df_train, df_val = train_test_split(
        df_rest,
        test_size=val_rel,
        stratify=df_rest['strata'],
        random_state=random_state
    )
    
    splits = {
        'train': df_train['route'].tolist(),
        'val':   df_val['route'].tolist(),
        'test':  df_test['route'].tolist()
    }
    
    # 5) para cada split, copia os arquivos CSV correspondentes
    for split_name, routes in splits.items():
        out_dir_split = os.path.join(output_dir, split_name)
        os.makedirs(out_dir_split, exist_ok=True)
        for r in routes:
            src = os.path.join(tracks_dir, r)
            dst = os.path.join(out_dir_split, r)
            if os.path.isfile(src):
                shutil.copy(src, dst)
            else:
                print(f"⚠️ Arquivo não encontrado: {src}")
        print(f"✅ {len(routes)} rotas copiadas para {out_dir_split}")
    
    # 6) salva os índices de cada split em CSV
    for split_name, routes in splits.items():
        pd.DataFrame({'route': routes}).to_csv(
            os.path.join(output_dir, f'{split_name}_routes.csv'),
            index=False
        )
    print(f"✅ Listas de rotas de cada split salvas em {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Divide estratificadamente as rotas e distribui em train/val/test"
    )
    parser.add_argument("tracks_dir",
                        help="Diretório contendo os CSVs das rotas já filtradas")
    parser.add_argument("summary_csv",
                        help="CSV de métricas (p.ex. analises/.../summary_metrics.csv)")
    parser.add_argument("output_dir",
                        help="Pasta base onde serão criadas subpastas train/, val/, test/")
    parser.add_argument("--stratify_col", default="duration_min",
                        help="Coluna numérica para estratificação (default: duration_min)")
    parser.add_argument("--n_bins", type=int, default=5,
                        help="Número de bins para estratos (default: 5)")
    parser.add_argument("--train_size", type=float, default=0.8,
                        help="Proporção para treino (default: 0.8)")
    parser.add_argument("--val_size", type=float, default=0.1,
                        help="Proporção para validação (default: 0.1)")
    parser.add_argument("--test_size", type=float, default=0.1,
                        help="Proporção para teste (default: 0.1)")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Semente para reprodutibilidade (default: 42)")
    args = parser.parse_args()
    
    stratified_split_and_move(
        tracks_dir=args.tracks_dir,
        summary_csv=args.summary_csv,
        output_dir=args.output_dir,
        stratify_col=args.stratify_col,
        n_bins=args.n_bins,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_state
    )
