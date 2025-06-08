#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import shutil
import argparse
import pandas as pd

def move_by_metrics(tracks_dir: str, output_dir: str, summary_csv: str,
                    param: str, min_val: float = None, max_val: float = None):
    """
    Lê summary_csv, filtra rotas cujo `param` está dentro de [min_val, max_val]
    e move os arquivos statevectors correspondentes de tracks_dir para output_dir.
    """
    # 1) Carrega o CSV de métricas
    df = pd.read_csv(summary_csv)
    if 'route' not in df.columns:
        raise ValueError(f"O CSV {summary_csv} não contém a coluna 'route'.")
    if param not in df.columns:
        raise ValueError(f"O CSV {summary_csv} não contém a coluna '{param}'.")

    # 2) Aplica filtros
    mask = pd.Series(True, index=df.index)
    if min_val is not None:
        mask &= df[param] >= min_val
    if max_val is not None:
        mask &= df[param] <= max_val

    to_move = df.loc[mask, 'route'].tolist()
    if not to_move:
        print("Nenhuma rota satisfaz o filtro.")
        return

    # 3) Move os arquivos
    os.makedirs(output_dir, exist_ok=True)
    moved, missing = [], []
    for fname in to_move:
        src = os.path.join(tracks_dir, fname)
        dst = os.path.join(output_dir, fname)
        if os.path.isfile(src):
            shutil.move(src, dst)
            moved.append(fname)
        else:
            missing.append(fname)

    # 4) Relatório
    print(f"\nParâmetro filtrado: {param}")
    if min_val is not None: print(f"  min >= {min_val}")
    if max_val is not None: print(f"  max <= {max_val}")
    print(f"\nArquivos movidos para '{output_dir}':")
    for f in moved:
        print(f"  - {f}")
    if missing:
        print(f"\nArquivos não encontrados em {tracks_dir}:")
        for f in missing:
            print(f"  - {f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Move rotas da pasta tracks/ filtrando pelo summary_metrics.csv"
    )
    parser.add_argument(
        "tracks_dir",
        help="Diretório de origem dos arquivos statevectors_*.csv"
    )
    parser.add_argument(
        "output_dir",
        help="Diretório de destino para onde mover as rotas filtradas"
    )
    parser.add_argument(
        "summary_csv",
        help="Caminho para summary_metrics.csv (p.ex.: analises/.../summary_metrics.csv)"
    )
    parser.add_argument(
        "param",
        help="Nome da coluna em summary_metrics.csv para usar no filtro"
    )
    parser.add_argument(
        "--min",
        dest="min_val",
        type=float,
        help="Valor mínimo (inclusive) para o parametro"
    )
    parser.add_argument(
        "--max",
        dest="max_val",
        type=float,
        help="Valor máximo (inclusive) para o parametro"
    )

    args = parser.parse_args()
    move_by_metrics(
        tracks_dir=args.tracks_dir,
        output_dir=args.output_dir,
        summary_csv=args.summary_csv,
        param=args.param,
        min_val=args.min_val,
        max_val=args.max_val
    )
