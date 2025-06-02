from eda_visualization import (
    load_track_intermediates,
    plot_all_routes,
    plot_route,
    plot_altitude_profile,
    plot_speed_profile,
    plot_histograms,
    plot_duration_distribution,
    plot_points_per_flight,
    plot_origin_destination,
    plot_route_colored_by_altitude,
    plot_route_colored_by_speed
)

# Exemplo de uso:
df = load_track_intermediates("output/track_with_intermediates.csv")

# 1) Visualizar todas as rotas (ou amostra de 50)
plot_all_routes(df, sample=50)

# 2) Visualizar rota específica
plot_route(df, flight_id="0e496e_1")

# 3) Perfil de altitude do voo
plot_altitude_profile(df, flight_id="0e496e_1")

# 4) Perfil de velocidade do voo
plot_speed_profile(df, flight_id="0e496e_1")

# 5) Histogramas gerais
plot_histograms(df)

# 6) Distribuição de duração de voo
plot_duration_distribution(df)

# 7) Pontos por voo
plot_points_per_flight(df)

# 8) Se existirem as colunas de aeroporto:
# plot_origin_destination(df)

# 9) Rota colorida por altitude e por velocidade
plot_route_colored_by_altitude(df, flight_id="0e496e_1")
plot_route_colored_by_speed(df, flight_id="0e496e_1")
