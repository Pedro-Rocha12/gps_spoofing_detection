# visualize_tracks.py
# Script para visualizar rotas de state vectors sobre um mapa interativo

import os
import glob
import random
import pandas as pd
import folium


def load_all_tracks(tracks_dir: str) -> dict:
    """
    Carrega todos os CSVs de state vectors do diretório.
    Retorna um dicionário {nome_arquivo: DataFrame}.
    """
    file_paths = glob.glob(os.path.join(tracks_dir, "statevectors_*.csv"))
    tracks = {}
    for fp in file_paths:
        df = pd.read_csv(fp)
        df = df.dropna(subset=['lat', 'lon'])  # descarta pontos sem coordenadas
        if df.empty:
            continue
        tracks[os.path.basename(fp)] = df
    return tracks


def plot_tracks(tracks_dfs: dict, map_center: list = None, zoom_start: int = 6) -> folium.Map:
    """
    Plota as rotas fornecidas em um mapa Folium.
    tracks_dfs: dict {nome_arquivo: DataFrame}
    Retorna o objeto folium.Map.
    """
    if map_center is None:
        all_lats = []
        all_lons = []
        for df in tracks_dfs.values():
            all_lats.extend(df['lat'].tolist())
            all_lons.extend(df['lon'].tolist())
        if not all_lats:
            raise ValueError("Nenhum ponto de coordenada disponível para plotar.")
        map_center = [sum(all_lats)/len(all_lats), sum(all_lons)/len(all_lons)]

    m = folium.Map(location=map_center, zoom_start=zoom_start)
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightblue', 'lightgreen', 'beige', 'darkblue']

    for idx, (fname, df) in enumerate(tracks_dfs.items()):
        coords = list(zip(df['lat'], df['lon']))
        folium.PolyLine(
            coords,
            color=colors[idx % len(colors)],
            weight=2.5,
            opacity=0.8,
            tooltip=fname
        ).add_to(m)
    return m


def plot_random_tracks(n: int, tracks_dir: str, output_html: str) -> folium.Map:
    """Plota n rotas aleatórias do diretório e salva em HTML."""
    tracks = load_all_tracks(tracks_dir)
    if n > len(tracks):
        raise ValueError(f"Foram solicitadas {n} rotas, mas apenas {len(tracks)} disponíveis.")
    selected = dict(random.sample(list(tracks.items()), n))
    m = plot_tracks(selected)
    _save_map(m, output_html)
    return m


def plot_selected_tracks(filenames: list, tracks_dir: str, output_html: str) -> folium.Map:
    """Plota rotas específicas fornecidas na lista e salva em HTML."""
    tracks = load_all_tracks(tracks_dir)
    selected = {f: tracks[f] for f in filenames if f in tracks}
    missing = set(filenames) - set(selected.keys())
    if missing:
        print(f"Atenção: os seguintes arquivos não foram encontrados: {missing}")
    m = plot_tracks(selected)
    _save_map(m, output_html)
    return m


def plot_all_tracks(tracks_dir: str, output_html: str) -> folium.Map:
    """Plota todas as rotas disponíveis e salva em HTML."""
    tracks = load_all_tracks(tracks_dir)
    m = plot_tracks(tracks)
    _save_map(m, output_html)
    return m


def _save_map(m: folium.Map, output_html: str):
    """Garante diretório e salva o mapa em HTML."""
    out_dir = os.path.dirname(output_html) or "visualizacoes"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, os.path.basename(output_html))
    m.save(path)
    print(f"Mapa salvo em: {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualizar rotas de state vectors em mapa interativo."
    )
    parser.add_argument(
        "tracks_dir",
        help="Diretório contendo os CSVs de state vectors."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--random",
        type=int,
        help="Número de rotas aleatórias a plotar."
    )
    group.add_argument(
        "--files",
        nargs='+',
        help="Lista de nomes de arquivos específicos para plotar."
    )
    group.add_argument(
        "--all",
        action='store_true',
        help="Plotar todas as rotas."
    )
    parser.add_argument(
        "--output",
        help="Nome do arquivo HTML de saída (será salvo em 'visualizacoes/')."
    )
    args = parser.parse_args()

    # Definir nome padrão se não informado
    if not args.output:
        if args.random:
            default_name = f"random{args.random}.html"
        elif args.files:
            default_name = "_".join(args.files) + ".html"
        else:
            default_name = "all.html"
        args.output = default_name

    if args.random:
        plot_random_tracks(args.random, args.tracks_dir, args.output)
    elif args.files:
        plot_selected_tracks(args.files, args.tracks_dir, args.output)
    elif args.all:
        plot_all_tracks(args.tracks_dir, args.output)
    else:
        parser.error("Nenhuma opção de plotagem fornecida.")
