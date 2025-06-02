# --------------------------------------------------------------------------------
# Nome do arquivo: eda_visualizations.py
# Objetivo: funções de visualização para dados “por ponto” com intermediates
#           (track_with_intermediates.csv). Permite inspeção de rotas individuais
#           ou todas as rotas, além de estatísticas gráficas.
# Dependências: pandas, matplotlib, numpy
# --------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_track_intermediates(csv_path: str) -> pd.DataFrame:
    """
    Carrega o CSV com colunas intermediárias (track_with_intermediates.csv).
    Retorna DataFrame com:
      ['icao24','callsign','time','lat','lon','baroaltitude','heading','onground',
       'flight_id','prev_lat','prev_lon','prev_time','prev_alt','prev_heading',
       'dist_m','delta_t','speed_emp','vert_rate_emp','delta_heading']
    """
    df = pd.read_csv(csv_path)
    return df


def plot_all_routes(df: pd.DataFrame, sample: int = None, figsize=(10, 6)):
    """
    Plota trajetórias (longitude x latitude) de todas as rotas presentes em df.
    Se `sample` for especificado, plota apenas esse número de rotas aleatórias.
    """
    unique_flights = df["flight_id"].unique()
    if sample is not None and sample < len(unique_flights):
        np.random.seed(42)
        chosen = np.random.choice(unique_flights, size=sample, replace=False)
    else:
        chosen = unique_flights

    plt.figure(figsize=figsize)
    for fid in chosen:
        sub = df[df["flight_id"] == fid]
        plt.plot(sub["lon"], sub["lat"], alpha=0.5, linewidth=0.8)

    plt.title("Trajetórias de Voo (todas ou amostra)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_route(df: pd.DataFrame, flight_id: str, figsize=(8, 5)):
    """
    Plota a trajetória (longitude x latitude) de um único flight_id.
    """
    sub = df[df["flight_id"] == flight_id]
    if sub.empty:
        print(f"⚠️ Nenhum dado encontrado para flight_id '{flight_id}'.")
        return

    plt.figure(figsize=figsize)
    plt.plot(sub["lon"], sub["lat"], marker="o", markersize=2, linewidth=1)
    plt.title(f"Trajetória do Voo: {flight_id}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_altitude_profile(df: pd.DataFrame, flight_id: str, figsize=(10, 4)):
    """
    Plota altitude barométrica ao longo do tempo para um único flight_id.
    """
    sub = df[df["flight_id"] == flight_id].copy()
    if sub.empty:
        print(f"⚠️ Nenhum dado encontrado para flight_id '{flight_id}'.")
        return

    sub["datetime"] = pd.to_datetime(sub["time"], unit="s")
    plt.figure(figsize=figsize)
    plt.plot(sub["datetime"], sub["baroaltitude"], color="#3D5A80")
    plt.title(f"Altitude ao longo do tempo — Voo {flight_id}")
    plt.xlabel("Tempo")
    plt.ylabel("Altitude barométrica (m)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_speed_profile(df: pd.DataFrame, flight_id: str, figsize=(10, 4)):
    """
    Plota velocidade empírica ao longo do tempo para um único flight_id.
    """
    sub = df[df["flight_id"] == flight_id].copy()
    if sub.empty:
        print(f"⚠️ Nenhum dado encontrado para flight_id '{flight_id}'.")
        return

    sub["datetime"] = pd.to_datetime(sub["time"], unit="s")
    plt.figure(figsize=figsize)
    plt.plot(sub["datetime"], sub["speed_emp"], color="#218380")
    plt.title(f"Velocidade ao longo do tempo — Voo {flight_id}")
    plt.xlabel("Tempo")
    plt.ylabel("Velocidade empírica (m/s)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_histograms(df: pd.DataFrame, figsize=(12, 8)):
    """
    Exibe histogramas gerais de distribuições:
      - speed_emp
      - vert_rate_emp
      - delta_heading
      - dist_m
      - baroaltitude
    """
    plt.figure(figsize=figsize)
    
    plt.subplot(2, 3, 1)
    plt.hist(df["speed_emp"].dropna(), bins=50, color="#218380", edgecolor="black")
    plt.title("Velocidade empírica (m/s)")
    plt.xlabel("m/s")

    plt.subplot(2, 3, 2)
    plt.hist(df["vert_rate_emp"].dropna(), bins=50, color="#2E86AB", edgecolor="black")
    plt.title("Taxa vertical empírica (m/s)")
    plt.xlabel("m/s")

    plt.subplot(2, 3, 3)
    plt.hist(df["delta_heading"].dropna(), bins=50, color="#A23B73", edgecolor="black")
    plt.title("Mudança de heading (graus/s)")
    plt.xlabel("graus/s")

    plt.subplot(2, 3, 4)
    plt.hist(df["dist_m"].dropna(), bins=50, color="#4C9F70", edgecolor="black")
    plt.title("Distância horizontal por segmento (m)")
    plt.xlabel("m")

    plt.subplot(2, 3, 5)
    plt.hist(df["baroaltitude"].dropna(), bins=50, color="#E07A5F", edgecolor="black")
    plt.title("Altitude barométrica (m)")
    plt.xlabel("m")

    plt.tight_layout()
    plt.show()


def plot_duration_distribution(df: pd.DataFrame, min_points: int = 2, figsize=(8, 4)):
    """
    Calcula e plota a distribuição das durações de voo (em minutos) para cada flight_id.
    Apenas considera voos com pelo menos `min_points` pontos.
    """
    grouped = df.groupby("flight_id")["time"].agg(["min", "max"])
    grouped["duration_min"] = (grouped["max"] - grouped["min"]) / 60.0
    grouped = grouped[grouped["duration_min"].notna()]

    plt.figure(figsize=figsize)
    plt.hist(grouped["duration_min"], bins=50, color="#E07A5F", edgecolor="black")
    plt.title("Distribuição de Duração de Voo (minutos)")
    plt.xlabel("Duração (min)")
    plt.ylabel("Número de Voos")
    plt.tight_layout()
    plt.show()


def plot_points_per_flight(df: pd.DataFrame, figsize=(8, 4)):
    """
    Calcula e plota a distribuição do número de pontos por flight_id.
    """
    pontos_por_voo = df.groupby("flight_id").size()
    plt.figure(figsize=figsize)
    plt.hist(pontos_por_voo.values, bins=50, color="#4C9F70", edgecolor="black")
    plt.title("Distribuição de Pontos por Voo")
    plt.xlabel("Número de pontos")
    plt.ylabel("Número de voos")
    plt.tight_layout()
    plt.show()


def plot_origin_destination(df: pd.DataFrame, figsize=(8, 4)):
    """
    Se existir coluna 'airport_origin' e 'airport_destination' no DataFrame, plota
    um diagrama de barras com as contagens de origem e destino.
    Caso contrário, imprime mensagem de alerta.
    """
    if "airport_origin" not in df.columns or "airport_destination" not in df.columns:
        print("⚠️ Colunas 'airport_origin' e/ou 'airport_destination' não encontradas no DataFrame.")
        return

    origin_counts = df["airport_origin"].value_counts()
    dest_counts = df["airport_destination"].value_counts()

    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    origin_counts.plot(kind="bar", color="#4C9F70", edgecolor="black")
    plt.title("Contagem de Aeroportos de Origem")
    plt.xlabel("ICAO Origem")
    plt.ylabel("Número de voos")

    plt.subplot(1, 2, 2)
    dest_counts.plot(kind="bar", color="#E07A5F", edgecolor="black")
    plt.title("Contagem de Aeroportos de Destino")
    plt.xlabel("ICAO Destino")
    plt.ylabel("Número de voos")

    plt.tight_layout()
    plt.show()


def plot_route_colored_by_altitude(df: pd.DataFrame, flight_id: str, figsize=(8, 6)):
    """
    Plota a trajetória de um voo colorida pela altitude barométrica.
    """
    sub = df[df["flight_id"] == flight_id]
    if sub.empty:
        print(f"⚠️ Nenhum dado encontrado para flight_id '{flight_id}'.")
        return

    sc = plt.scatter(sub["lon"], sub["lat"], c=sub["baroaltitude"], cmap="viridis", s=4)
    plt.colorbar(sc, label="Altitude (m)")
    plt.title(f"Trajetória colorida por altitude — Voo {flight_id}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_route_colored_by_speed(df: pd.DataFrame, flight_id: str, figsize=(8, 6)):
    """
    Plota a trajetória de um voo colorida pela velocidade empírica.
    """
    sub = df[df["flight_id"] == flight_id]
    if sub.empty:
        print(f"⚠️ Nenhum dado encontrado para flight_id '{flight_id}'.")
        return

    sc = plt.scatter(sub["lon"], sub["lat"], c=sub["speed_emp"], cmap="plasma", s=4)
    plt.colorbar(sc, label="Velocidade (m/s)")
    plt.title(f"Trajetória colorida por velocidade — Voo {flight_id}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
