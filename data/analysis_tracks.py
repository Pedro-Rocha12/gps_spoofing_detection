# analysis_tracks.py
# Script para análise qualitativa e quantitativa de rotas de state vectors
import os
import glob
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def haversine(lat1, lon1, lat2, lon2):
    """Calcula distância em km entre dois pontos usando fórmula de Haversine."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_tracks(tracks_dir: str, selected: list = None, random_n: int = None) -> dict:
    """Carrega tracks de CSVs, converte colunas para numérico e permite seleção por nome ou aleatória."""
    paths = glob.glob(os.path.join(tracks_dir, "statevectors_*.csv"))
    if selected:
        logger.info("Filtrando para %d rotas específicas", len(selected))
        paths = [p for p in paths if os.path.basename(p) in selected]
    tracks = {}
    for p in paths:
        name = os.path.basename(p)
        logger.debug("Carregando %s", name)
        df = pd.read_csv(p)
        # Converter lat, lon e velocity para numérico
        for col in ['lat', 'lon', 'velocity']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        # Converter time e filtrar
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'].astype(int), unit='s', utc=True)
        df = df.dropna(subset=['lat', 'lon', 'velocity'])
        if df.empty:
            logger.warning("Nenhum dado válido em %s, pulando.", name)
            continue
        tracks[name] = df
    if random_n:
        keys = list(tracks.keys())
        if random_n > len(keys):
            msg = f"Solicitadas {random_n} rotas, mas só existem {len(keys)} disponíveis."
            logger.error(msg)
            raise ValueError(msg)
        sampled = np.random.choice(keys, random_n, replace=False)
        logger.info("Selecionando %d rotas aleatórias", random_n)
        tracks = {k: tracks[k] for k in sampled}
    logger.info("Total de rotas carregadas: %d", len(tracks))
    return tracks


def compute_metrics(tracks: dict) -> pd.DataFrame:
    """Calcula métricas: duração, distância, velocidades, acelerações, curvas e taxas verticais."""
    records = []
    for name, df in tracks.items():
        logger.debug("Processando rota %s", name)
        times = df['time']
        duration = (times.iloc[-1] - times.iloc[0]).total_seconds() / 60 if len(times) > 1 else 0
        if len(df) > 1:
            dist = haversine(df['lat'].values[:-1], df['lon'].values[:-1],
                             df['lat'].values[1:], df['lon'].values[1:])
            total_dist = float(dist.sum())
        else:
            total_dist = 0.0
        avg_speed = float(df['velocity'].mean() * 3.6)
        max_speed = float(df['velocity'].max() * 3.6)
        if len(df) > 1:
            speed = df['velocity'].to_numpy()
            dt = np.diff(df['time'].astype(np.int64) / 1e9)
            accel = np.diff(speed) / dt
            avg_accel = float(np.nanmean(accel))
            max_accel = float(np.nanmax(accel))
        else:
            avg_accel, max_accel = np.nan, np.nan
        if 'heading' in df.columns and len(df) > 2:
            hdg = df['heading'].to_numpy()
            dh = np.abs((hdg[1:] - hdg[:-1] + 180) % 360 - 180)
            num_turns = int(np.sum(dh > 10))
            avg_turn = float(np.nanmean(dh))
        else:
            num_turns, avg_turn = np.nan, np.nan
        if 'vertrate' in df.columns:
            vr = df['vertrate'].dropna().to_numpy()
            avg_vr = float(np.mean(vr))
            max_climb = float(np.max(vr))
            max_desc = float(np.min(vr))
        else:
            avg_vr, max_climb, max_desc = np.nan, np.nan, np.nan
        records.append({
            'route': name,
            'duration_min': duration,
            'distance_km': total_dist,
            'avg_speed_kmh': avg_speed,
            'max_speed_kmh': max_speed,
            'avg_accel_m_s2': avg_accel,
            'max_accel_m_s2': max_accel,
            'num_turns': num_turns,
            'avg_turn_deg': avg_turn,
            'avg_vert_rate_m_s': avg_vr,
            'max_climb_m_s': max_climb,
            'max_descend_m_s': max_desc
        })
        logger.info("Rota %s: distância = %.2f km, duração = %.2f min", name, total_dist, duration)
    return pd.DataFrame(records)


def plot_and_save_histograms(summary: pd.DataFrame, out_dir: str):
    """Gera histogramas com linha de média/mediana e salva em PNG."""
    metrics = {
        'distance_km': 'Distância (km)',
        'duration_min': 'Duração (min)',
        'avg_speed_kmh': 'Velocidade Média (km/h)',
        'max_speed_kmh': 'Velocidade Máxima (km/h)',
        'avg_accel_m_s2': 'Aceleração Média (m/s²)',
        'num_turns': 'Número de Curvas',
        'avg_vert_rate_m_s': 'Taxa Vertical Média (m/s)'
    }
    for key, label in metrics.items():
        data = summary[key].dropna()
        if data.empty:
            logger.debug("Nao ha dados para histograma de %s", key)
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(data, bins=20, edgecolor='black')
        mean, med = data.mean(), data.median()
        ax.axvline(mean, linestyle='--', linewidth=1, color='grey', label=f'Média = {mean:.2f}')
        ax.axvline(med, linestyle=':', linewidth=1, color='black', label=f'Mediana = {med:.2f}')
        ax.set_title(f'Distribuição de {label}')
        ax.set_xlabel(label)
        ax.set_ylabel('Frequência')
        ax.legend()
        ax.grid(True, linestyle=':', linewidth=0.5)
        fig.tight_layout()
        path = os.path.join(out_dir, f'hist_{key}.png')
        fig.savefig(path)
        plt.close(fig)
        logger.info("Histograma %s salvo em %s", key, path)


def plot_and_save_scatter(summary: pd.DataFrame, x: str, y: str, out_dir: str):
    """Gera scatter plot com coeficiente de correlação e salva em PNG."""
    data = summary[[x, y]].dropna()
    if len(data) < 2:
        logger.debug("Dados insuficientes para scatter %s vs %s", x, y)
        return
    std_x, std_y = data[x].std(), data[y].std()
    corr = data[x].corr(data[y]) if std_x and std_y else np.nan
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(data[x], data[y], alpha=0.7)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f'{y} vs {x} (r = {corr:.2f})')
    ax.grid(True, linestyle=':', linewidth=0.5)
    fig.tight_layout()
    path = os.path.join(out_dir, f'scatter_{x}_{y}.png')
    fig.savefig(path)
    plt.close(fig)
    logger.info("Scatter %s vs %s salvo em %s", x, y, path)


def main(tracks_dir: str, routes: list = None, random_n: int = None):
    logger.info("Iniciando análise de rotas em %s", tracks_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    label = 'all' if not routes and not random_n else (
        'sel_' + '_'.join(routes) if routes else f'rand{random_n}'
    )
    out_dir = os.path.join('analises', f'{label}_{timestamp}')
    os.makedirs(out_dir, exist_ok=True)
    tracks = load_tracks(tracks_dir, selected=routes, random_n=random_n)
    if not tracks:
        logger.error('Nenhuma rota para análise.')
        return
    summary = compute_metrics(tracks)
    csv_path = os.path.join(out_dir, 'summary_metrics.csv')
    summary.to_csv(csv_path, index=False)
    logger.info("Resumo salvo em %s", csv_path)
    plot_and_save_histograms(summary, out_dir)
    plot_and_save_scatter(summary, 'distance_km', 'avg_speed_kmh', out_dir)
    plot_and_save_scatter(summary, 'num_turns', 'max_accel_m_s2', out_dir)
    logger.info("Análise concluída. Resultados em %s", out_dir)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Análise de rotas com logging aprimorado')
    parser.add_argument('tracks_dir', help='Diretório com CSVs de state vectors')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--routes', nargs='+', help='Lista de arquivos para analisar')
    group.add_argument('--random', type=int, help='Quantas rotas aleatórias analisar')
    args = parser.parse_args()
    main(args.tracks_dir, routes=args.routes, random_n=args.random)
