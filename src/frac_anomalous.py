import pandas as pd
import matplotlib.pyplot as plt

# 1. Carrega o CSV que você acabou de gerar
df = pd.read_csv("analysis/route_classification.csv")

# 2. Estatísticas descritivas
print(df["frac_anomalous"].describe())

# 3. Histograma
plt.figure(figsize=(8,4))
plt.hist(df["frac_anomalous"], bins=50, edgecolor="black")
plt.title("Distribuição da fração de pontos anômalos por rota")
plt.xlabel("Fração de pontos anômalos")
plt.ylabel("Número de rotas")
plt.grid(True, linestyle=":")
plt.tight_layout()
plt.show()
