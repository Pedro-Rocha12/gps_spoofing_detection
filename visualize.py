import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_altitude_and_velocity(df: pd.DataFrame, callsign: str):
    """Plota altitude e velocidade ao longo do tempo para uma aeronave específica."""
    # Seleciona todos os voos com o mesmo callsign (podem haver mais de um icao24)
    data = df[df["callsign"] == callsign].copy()

    if data.empty:
        print(f"Nenhum dado encontrado para o callsign '{callsign}'.")
        return

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel("Tempo")
    ax1.set_ylabel("Altitude (m)", color="tab:blue")
    ax1.plot(data["datetime"], data["baroaltitude"], label="Altitude", color="tab:blue")
    ax1.tick_params(axis='y', labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Velocidade (m/s)", color="tab:red")
    ax2.plot(data["datetime"], data["velocity"], label="Velocidade", color="tab:red")
    ax2.tick_params(axis='y', labelcolor="tab:red")

    fig.tight_layout()
    plt.title(f"Altitude e Velocidade — Callsign: {callsign}")
    plt.show()

def plot_distance_over_time(df: pd.DataFrame, callsign: str):
    """Plota a distância acumulada ao longo do tempo para um callsign."""
    data = df[df["callsign"] == callsign].copy()
    if data.empty or "distance_m" not in data.columns:
        print(f"Nenhum dado de distância encontrado para '{callsign}'.")
        return

    data["dist_acumulada_km"] = data["distance_m"].cumsum() / 1000

    plt.figure(figsize=(12, 6))
    plt.plot(data["datetime"], data["dist_acumulada_km"], color="green")
    plt.title(f"Distância acumulada — Callsign: {callsign}")
    plt.xlabel("Tempo")
    plt.ylabel("Distância (km)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_tracks(csv_path, flight_ids=None, max_flights=5):
    """
    Plota trajetórias (latitude vs longitude) dos voos a partir do CSV de track.

    Parâmetros:
    - csv_path: caminho para o CSV com colunas ['icao24', 'lat', 'lon', ...].
    - flight_ids: lista de ICAO24 para visualizar; se None, usa até max_flights primeiros.
    - max_flights: número máximo de voos a plotar quando flight_ids for None.
    """
    # Carregar dados
    df = pd.read_csv(csv_path)
    unique_ids = df['icao24'].unique()

    if flight_ids:
        # Filtra apenas os válidos
        flight_ids = [f for f in flight_ids if f in unique_ids]
        if not flight_ids:
            print("Nenhum ICAO24 válido encontrado no CSV.")
            return
    else:
        flight_ids = unique_ids[:max_flights]

    # Definir projeção PlateCarree para plotagem geográfica
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=proj)

    # Adicionar elementos de mapa
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)

    # Determinar extensão do mapa com base nos dados
    lons = df['lon']
    lats = df['lat']
    margin = 1.0  # graus de margem
    ax.set_extent([
        lons.min() - margin, lons.max() + margin,
        lats.min() - margin, lats.max() + margin
    ], crs=proj)

    # Plotar cada trajetória
    for icao in flight_ids:
        data = df[df['icao24'] == icao]
        ax.plot(
            data['lon'], data['lat'],
            transform=ccrs.Geodetic(),
            label=icao
        )

    ax.set_title('Trajetórias de Voo sobre Mapa Real')
    ax.legend(title='ICAO24', loc='upper right')
    plt.show()(csv_path, flight_ids=None, max_flights=5)
    """
    Plota trajetórias (latitude vs longitude) dos voos a partir do CSV de track.

    Parâmetros:
    - csv_path: caminho para o CSV com colunas ['icao24', 'lat', 'lon', ...].
    - flight_ids: lista de ICAO24 para visualizar; se None, usa até max_flights primeiros.
    - max_flights: número máximo de voos a plotar quando flight_ids for None.
    """
    # Carregar dados
    df = pd.read_csv(csv_path)
    unique_ids = df['icao24'].unique()

    if flight_ids:
        # Filtra apenas os válidos
        flight_ids = [f for f in flight_ids if f in unique_ids]
        if not flight_ids:
            print("Nenhum ICAO24 válido encontrado no CSV.")
            return
    else:
        flight_ids = unique_ids[:max_flights]

    plt.figure(figsize=(10, 8))
    for icao in flight_ids:
        data = df[df['icao24'] == icao]
        plt.plot(data['lon'], data['lat'], label=icao)

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Trajetórias de Voo')
    plt.legend(title='ICAO24')
    plt.grid(True)
    plt.tight_layout()
    plt.show()