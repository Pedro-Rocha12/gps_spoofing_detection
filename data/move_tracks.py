# move_tracks.py
# Script para mover arquivos de rota da pasta tracks para um diretório definido pelo usuário

import os
import shutil
import argparse
import glob

def move_tracks(tracks_dir: str, output_dir: str, filenames: list):
    """
    Move os arquivos especificados da pasta tracks_dir para output_dir.
    filenames: lista de nomes de arquivos (ex: ['statevectors_abc.csv', ...])
    """
    # Certifica que o diretório de saída existe
    os.makedirs(output_dir, exist_ok=True)

    # Se o usuário passar '*', considerar todos os arquivos statevectors_*.csv
    all_files = glob.glob(os.path.join(tracks_dir, 'statevectors_*.csv'))
    if filenames == ['*']:
        to_move = [os.path.basename(p) for p in all_files]
    else:
        to_move = filenames

    moved = []
    not_found = []
    for fname in to_move:
        src = os.path.join(tracks_dir, fname)
        if os.path.exists(src):
            dst = os.path.join(output_dir, fname)
            shutil.move(src, dst)
            moved.append(fname)
        else:
            not_found.append(fname)

    # Relatório
    print(f"Arquivos movidos para '{output_dir}':")
    for f in moved:
        print(f"  - {f}")
    if not_found:
        print("\nArquivos não encontrados:")
        for f in not_found:
            print(f"  - {f}")


def main():
    parser = argparse.ArgumentParser(
        description='Move arquivos de statevectors da pasta tracks para outro diretório.'
    )
    parser.add_argument(
        'tracks_dir',
        help='Diretório de origem contendo os arquivos statevectors_*.csv'
    )
    parser.add_argument(
        'output_dir',
        help='Diretório de destino onde os arquivos serão movidos'
    )
    parser.add_argument(
        '--files', '-f',
        nargs='+',
        required=True,
        help="Lista de nomes de arquivos para mover ou '*' para todos."
    )
    args = parser.parse_args()
    move_tracks(args.tracks_dir, args.output_dir, args.files)

if __name__ == '__main__':
    main()
