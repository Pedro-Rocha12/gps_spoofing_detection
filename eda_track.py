# --------------------------------------------------------------------------------
# Nome do arquivo: eda_track_2.py
# Objetivo: EDA + limpeza + features “por ponto” + features “por voo” do CSV de tracks segmentado
#           (gera também um CSV separado com todos os voos excluídos pela limpeza).
# Dependências: pandas, numpy, matplotlib
# --------------------------------------------------------------------------------

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def haversine_np(lat1, lon1, lat2, lon2):
    """
    Calcula a distância haversine (em metros) entre dois vetores de coordenadas (lat1, lon1) e (lat2, lon2).
    Todos os arrays devem estar no mesmo comprimento ou ser escalar/serie compatível.
    """
    R = 6371e3  # raio da Terra em metros
    φ1 = np.radians(lat1)
    φ2 = np.radians(lat2)
    Δφ = np.radians(lat2 - lat1)
    Δλ = np.radians(lon2 - lon1)
    a = np.sin(Δφ / 2.0) ** 2 + np.cos(φ1) * np.cos(φ2) * np.sin(Δλ / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def compute_intermediate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe DataFrame contendo ao menos as colunas:
      ['icao24','callsign','time','lat','lon','baroaltitude','heading','onground','flight_id']
    Ordena por (flight_id, time) e cria as colunas:
      prev_lat, prev_lon, prev_time, prev_alt, prev_heading,
      dist_m, delta_t, speed_emp, vert_rate_emp, delta_heading
    Retorna o DataFrame estendido.
    """
    # 1) Ordenar por flight_id e time
    df = df.sort_values(by=["flight_id", "time"]).reset_index(drop=True)

    # 2) Criar colunas “anteriores” dentro de cada flight_id
    df[["prev_lat", "prev_lon", "prev_time", "prev_alt", "prev_heading"]] = (
        df
        .groupby("flight_id", sort=False)[["lat", "lon", "time", "baroaltitude", "heading"]]
        .shift(1)
    )

    # 3) Distância horizontal (m) – haversine entre (prev_lat, prev_lon) e (lat, lon)
    mask_prev = df["prev_lat"].notna()
    df.loc[mask_prev, "dist_m"] = haversine_np(
        df.loc[mask_prev, "prev_lat"].astype(float),
        df.loc[mask_prev, "prev_lon"].astype(float),
        df.loc[mask_prev, "lat"].astype(float),
        df.loc[mask_prev, "lon"].astype(float),
    )
    df.loc[~mask_prev, "dist_m"] = 0.0

    # 4) Delta de tempo (s)
    df.loc[mask_prev, "delta_t"] = df.loc[mask_prev, "time"].astype(float) - df.loc[mask_prev, "prev_time"].astype(float)
    df.loc[~mask_prev, "delta_t"] = np.nan

    # 5) Velocidade empírica (m/s)
    df.loc[mask_prev, "speed_emp"] = df.loc[mask_prev, "dist_m"] / df.loc[mask_prev, "delta_t"]
    df.loc[~mask_prev, "speed_emp"] = 0.0

    # 6) Taxa vertical empírica (m/s)
    df.loc[mask_prev, "vert_rate_emp"] = (
        df.loc[mask_prev, "baroaltitude"].astype(float) - df.loc[mask_prev, "prev_alt"].astype(float)
    ) / df.loc[mask_prev, "delta_t"]
    df.loc[~mask_prev, "vert_rate_emp"] = 0.0

    # 7) Mudança de heading (graus/s) com ajuste para principal valor em [-180, 180]
    df.loc[mask_prev, "delta_heading"] = df.loc[mask_prev, "heading"].astype(float) - df.loc[mask_prev, "prev_heading"].astype(float)
    # Corrige wrap‐around (por exemplo, de 350° para 10° resulta em -340°, mas queremos +20°, etc.)
    df["delta_heading"] = ((df["delta_heading"] + 180) % 360) - 180

    return df

def build_flight_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe DataFrame “por ponto” (já limpo e com colunas intermediárias):
      ['icao24','callsign','time','lat','lon','baroaltitude','heading','onground','flight_id',
       'prev_lat','prev_lon','prev_time','prev_alt','prev_heading','dist_m','delta_t',
       'speed_emp','vert_rate_emp','delta_heading']
    e retorna DataFrame “por voo” (cada linha é um flight_id), com features agregadas:
      - n_points           : número de pontos de track
      - duration_sec       : duração total do voo (s)
      - mean_speed         : velocidade empírica média (m/s)
      - max_speed          : velocidade empírica máxima (m/s)
      - std_speed          : desvio‐padrão da velocidade empírica (m/s)
      - max_vert_rate      : maior taxa vertical empírica (m/s)
      - min_vert_rate      : menor taxa vertical empírica (m/s)
      - mean_vert_rate     : taxa vertical média (m/s)
      - max_delta_heading  : maior variação absoluta de heading (graus/s)
      - std_delta_heading  : desvio‐padrão da variação de heading (graus/s)
      - total_horizontal   : soma de todas as distâncias horizontais (m)
      - mean_altitude      : altitude barométrica média (m)
      - max_altitude       : altitude barométrica máxima (m)
      - min_altitude       : altitude barométrica mínima (m)
      - std_altitude       : desvio‐padrão da altitude (m)
      - airport_origin     : ICAO do aeroporto de origem (coluna deixada vazia aqui)
      - airport_destination: ICAO do aeroporto de destino (coluna deixada vazia aqui)
    """
    # 1) Número de pontos
    agg_npoints = df.groupby("flight_id").size().rename("n_points")

    # 2) Duração (s)
    times = df.groupby("flight_id")["time"].agg(["min", "max"])
    times["duration_sec"] = times["max"] - times["min"]

    # 3) Estatísticas de speed_emp
    spd = df.groupby("flight_id")["speed_emp"].agg(
        mean_speed="mean",
        max_speed="max",
        std_speed="std"
    )

    # 4) Estatísticas de vert_rate_emp
    vr = df.groupby("flight_id")["vert_rate_emp"].agg(
        max_vert_rate="max",
        min_vert_rate="min",
        mean_vert_rate="mean"
    )

    # 5) Estatísticas de delta_heading (usar valor absoluto)
    dh = df.groupby("flight_id")["delta_heading"].agg(
        max_delta_heading=lambda x: x.abs().max(),
        std_delta_heading="std"
    )

    # 6) Distância total horizontal (m)
    total_hor = df.groupby("flight_id")["dist_m"].sum().rename("total_horizontal")

    # 7) Estatísticas de altitude
    alt = df.groupby("flight_id")["baroaltitude"].agg(
        mean_altitude="mean",
        max_altitude="max",
        min_altitude="min",
        std_altitude="std"
    )

    # 8) Colunas “origem” e “destino” (vamos deixar vazias, pois a rota
    #    já está definida em outro CSV de metadados)
    airport_origin = pd.Series(index=agg_npoints.index, dtype="object", name="airport_origin")
    airport_destination = pd.Series(index=agg_npoints.index, dtype="object", name="airport_destination")

    # Concatena tudo:
    flight_feats = pd.concat(
        [
            agg_npoints,
            times["duration_sec"],
            spd,
            vr,
            dh,
            total_hor,
            alt,
            airport_origin,
            airport_destination
        ],
        axis=1
    )

    # Preenche NaN gerados pelos std com 0
    flight_feats["std_speed"] = flight_feats["std_speed"].fillna(0)
    flight_feats["std_altitude"] = flight_feats["std_altitude"].fillna(0)
    flight_feats["std_delta_heading"] = flight_feats["std_delta_heading"].fillna(0)

    return flight_feats

def run_eda(track_csv_path: str, apply_clean: bool = True):
    """
    1) Lê o CSV segmentado (com coluna 'flight_id')
    2) (Opcional) Limpeza inicial:
         - remove onground, altitudes negativas, registros sem heading
         - descarta duplicatas exatas (flight_id, time, lat, lon)
         - descarta voos com menos de 2 pontos válidos
    3) Descarta voos com duração < 20 minutos ou > 2 horas
    4) Calcula colunas intermediárias (dist_m, delta_t, speed_emp, vert_rate_emp, delta_heading)
    5) Filtra outliers físicos (speed_emp > 350 m/s, |vert_rate_emp| > 30 m/s)
    6) Gera:
         (A) CSV “por ponto” com colunas intermediárias → output/track_with_intermediates.csv  
         (B) CSV “com voos totalmente excluídos” → output/excluded_flights.csv  
         (C) CSV “por voo” (flight_level_features) → output/flight_level_features.csv  
    7) Mostra estatísticas e alguns gráficos
    """

    # ------------------------------
    # 1) Carregar dados originais
    # ------------------------------
    print("🔄 Carregando CSV de track segmentado em memória…")
    df_orig = pd.read_csv(track_csv_path)

    # Copia para limpeza
    df = df_orig.copy()

    # ------------------------------
    # 2) Limpeza inicial (remover pontos inválidos)
    # ------------------------------
    if apply_clean:
        # 2.1) Remover pontos com onground=True
        df = df[~df["onground"]].copy()

        # 2.2) Remover altitudes negativas
        df = df[df["baroaltitude"] >= 0].copy()

        # 2.3) Descartar pontos sem heading
        df = df.dropna(subset=["heading"])

        # 2.4) Ordenar e eliminar duplicatas exatas
        df = df.sort_values(["flight_id", "time"])
        df = df.drop_duplicates(subset=["flight_id", "time", "lat", "lon"])

        # 2.5) Descartar voos com menos de 2 pontos após as etapas acima
        counts = df["flight_id"].value_counts()
        voos_com_2pts = counts[counts >= 2].index
        df = df[df["flight_id"].isin(voos_com_2pts)].copy()
    else:
        counts = df["flight_id"].value_counts()
        voos_com_2pts = counts[counts >= 2].index

    # ------------------------------
    # 3) Descartar voos com duração < 30 min ou > 4 horas
    # ------------------------------
    # Primeiro, calcular duração de cada voo no df atual:
    voo_times = df.groupby("flight_id")["time"].agg(["min", "max"]).copy()
    voo_times["duration_sec"] = voo_times["max"] - voo_times["min"]

    # Identificar flight_ids com duração fora do intervalo desejado:
    dur_menor_30 = voo_times[voo_times["duration_sec"] < 30 * 60].index.tolist()
    dur_maior_4h = voo_times[voo_times["duration_sec"] > 4 * 3600].index.tolist()
    dur_excluidos_ids = set(dur_menor_30 + dur_maior_4h)

    # Agora, remova esses voos de df:
    df = df[~df["flight_id"].isin(dur_excluidos_ids)].copy()

    # ------------------------------
    # 4) Identificar, no DataFrame original, todos os registros cujos flight_id foram excluídos
    # ------------------------------
    set_orig = set(df_orig["flight_id"].unique())
    set_limpo = set(df["flight_id"].unique())
    flightids_excluidos = list(set_orig - set_limpo)

    # Salvar todos esses registros originais em CSV (voos totalmente removidos)
    if flightids_excluidos:
        df_excluidos = df_orig[df_orig["flight_id"].isin(flightids_excluidos)].copy()
        output_excluidos_dir = "output"
        os.makedirs(output_excluidos_dir, exist_ok=True)
        path_excluidos = os.path.join(output_excluidos_dir, "excluded_flights.csv")
        df_excluidos.to_csv(path_excluidos, index=False)
        print(f"\n✅ CSV com voos completamente excluídos salvo em: {path_excluidos}")
    else:
        print("\n⚠️ Nenhum voo foi totalmente excluído nesta etapa.")

    # ------------------------------
    # 5) Cálculo de colunas intermediárias (distância, velocidade etc.)
    # ------------------------------
    df = compute_intermediate_columns(df)

    # 5.1) Descartar registros onde delta_t <= 0 ou NaN (primeiro ponto de cada voo)
    df = df[df["delta_t"].notna() & (df["delta_t"] > 0)].copy()

    # 5.2) Filtrar outliers físicos
    df = df[df["speed_emp"] <= 350.0]
    df = df[df["vert_rate_emp"].abs() <= 30.0]
    # (A variação de heading em si não está sendo filtrada para não remover manobras reais.)

    # ------------------------------
    # 6) Salvar CSV “por ponto” com colunas intermediárias
    # ------------------------------
    ponto_output_dir = "output"
    os.makedirs(ponto_output_dir, exist_ok=True)
    ponto_output_path = os.path.join(ponto_output_dir, "track_with_intermediates.csv")
    df.to_csv(ponto_output_path, index=False)
    print(f"\n✅ CSV “por ponto” com colunas intermediárias salvo em: {ponto_output_path}")

    # ------------------------------
    # 7) Mostrar estatísticas básicas
    # ------------------------------
    print("\n===== Dados após limpeza e cálculos (head) =====")
    print(df.head().to_string(index=False))
    print("\n===== Estrutura do DataFrame =====")
    print(df.info())

    num_voos = df["flight_id"].nunique()
    num_aeronaves = df["icao24"].nunique()
    print(f"\n🔢 Número de voos únicos (flight_id): {num_voos}")
    print(f"🛩  Número de aeronaves distintas (icao24): {num_aeronaves}")

    cols_base = ["time", "lat", "lon", "baroaltitude", "heading", "speed_emp", "vert_rate_emp", "delta_heading"]
    desc = df[cols_base].describe()
    print("\n===== Estatísticas descritivas (time, lat, lon, baroaltitude, heading + speed_emp, vert_rate_emp, delta_heading) =====")
    print(desc)

    # 7.1) Pontos por voo
    pontos_por_voo = df.groupby("flight_id").size()
    print("\n===== Estatísticas de Pontos por Voo =====")
    print(pontos_por_voo.describe())

    # 7.2) Duração de cada voo (time)
    voo_times_limpo = df.groupby("flight_id")["time"].agg(["min", "max"]).copy()
    voo_times_limpo["duration_sec"] = voo_times_limpo["max"] - voo_times_limpo["min"]
    print("\n===== Estatísticas de Duração de Voo (segundos) =====")
    print(voo_times_limpo["duration_sec"].describe())

    # ------------------------------
    # 8) Gráficos
    # ------------------------------
    # 8.1) Histograma de pontos por voo
    plt.figure(figsize=(8, 4))
    plt.hist(pontos_por_voo.values, bins=50, color="#4C9F70", edgecolor="black")
    plt.title("Distribuição de Pontos por Voo (flight_id)")
    plt.xlabel("Número de pontos no registro de track")
    plt.ylabel("Contagem de voos")
    plt.tight_layout()
    plt.show()

    # 8.2) Histograma de duração (minutos)
    plt.figure(figsize=(8, 4))
    plt.hist(voo_times_limpo["duration_sec"].values / 60, bins=50, color="#E07A5F", edgecolor="black")
    plt.title("Distribuição de Durações de Voo (minutos)")
    plt.xlabel("Duração do voo (minutos)")
    plt.ylabel("Contagem de voos")
    plt.tight_layout()
    plt.show()

    # 8.3) Histograma de velocidade empírica
    plt.figure(figsize=(8, 4))
    plt.hist(df["speed_emp"].dropna().values, bins=50, color="#218380", edgecolor="black")
    plt.title("Distribuição de Velocidades Empíricas (m/s)")
    plt.xlabel("Velocidade empírica (m/s)")
    plt.ylabel("Frequência")
    plt.tight_layout()
    plt.show()

    # 8.4) Histograma de taxa vertical empírica
    plt.figure(figsize=(8, 4))
    plt.hist(df["vert_rate_emp"].dropna().values, bins=50, color="#2E86AB", edgecolor="black")
    plt.title("Distribuição de Taxas Verticais Empíricas (m/s)")
    plt.xlabel("Taxa vertical empírica (m/s)")
    plt.ylabel("Frequência")
    plt.tight_layout()
    plt.show()

    # 8.5) Histograma de delta_heading
    plt.figure(figsize=(8, 4))
    plt.hist(df["delta_heading"].dropna().values, bins=50, color="#A23B73", edgecolor="black")
    plt.title("Distribuição de Mudança de Heading (graus/s)")
    plt.xlabel("Mudança de heading por segundo")
    plt.ylabel("Frequência")
    plt.tight_layout()
    plt.show()

    # 8.6) Trajetórias de exemplo: top-5 voos com mais pontos
    top5 = pontos_por_voo.sort_values(ascending=False).head(5).index.tolist()
    plt.figure(figsize=(10, 6))
    for fid in top5:
        sub = df[df["flight_id"] == fid]
        plt.plot(sub["lon"], sub["lat"], label=fid, alpha=0.8)
    plt.title("Trajetórias de Exemplo (Top-5 voos por número de pontos)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(title="flight_id", fontsize="small", loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 8.7) Altitude ao longo do tempo para o voo de maior número de pontos
    exemplo = top5[0]
    sub_ex = df[df["flight_id"] == exemplo].copy()
    sub_ex["datetime"] = pd.to_datetime(sub_ex["time"], unit="s")
    plt.figure(figsize=(10, 4))
    plt.plot(sub_ex["datetime"], sub_ex["baroaltitude"], color="#3D5A80")
    plt.title(f"Altitude ao longo do tempo — voo {exemplo}")
    plt.xlabel("Tempo")
    plt.ylabel("Altitude barométrica (m)")
    plt.tight_layout()
    plt.show()

    # ------------------------------
    # 9) Build “por voo” features e salvar CSV final
    # ------------------------------
    df_flight_feats = build_flight_features(df)

    print("\n===== Exemplo de DataFrame Agregado (“por voo”) =====")
    print(df_flight_feats.head().to_string())

    output_feats_path = os.path.join(ponto_output_dir, "flight_level_features.csv")
    df_flight_feats.to_csv(output_feats_path, index=True)
    print(f"\n✅ CSV “por voo” (flight_level_features) salvo em: {output_feats_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python eda_track_2.py <caminho_para_track_segmentado_csv>")
        sys.exit(1)

    track_seg_path = sys.argv[1]
    run_eda(track_seg_path, apply_clean=True)
