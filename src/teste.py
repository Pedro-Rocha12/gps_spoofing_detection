import pandas as pd

# 1) Carrega o CSV com frac_anomalous (já gerado pelo seu classify_routes.py)
df = pd.read_csv("analysis/route_classification_dynamic.csv")

# 2) Calcula quantis de interesse
q90 = df["frac_anomalous"].quantile(0.90)
q95 = df["frac_anomalous"].quantile(0.95)
q98 = df["frac_anomalous"].quantile(0.98)
print(f"90ᵒ percentil: {q90:.4f}")
print(f"95ᵒ percentil: {q95:.4f}")
print(f"98ᵒ percentil: {q98:.4f}")

# 3) Veja quantas rotas cada limiar marcaria
for q,name in [(q90,"P90"),(q95,"P95"),(q98,"P98")]:
    n = (df["frac_anomalous"] >= q).sum()
    print(f"{name}: marcaria {n} rotas de {len(df)}")

# 4) Agora redefina is_spoof com base, por exemplo, no 95ᵒ percentil
dynamic_thresh = q95
df["is_spoof_dynamic"] = df["frac_anomalous"] >= dynamic_thresh
df.to_csv("analysis/route_classification_dynamic.csv", index=False)
print(f"\n✅ Salvo classification com limiar dinâmico={dynamic_thresh:.4f}")


df = pd.read_csv("analysis/route_classification_dynamic.csv")
spoofed = df[df["is_spoof_dynamic"]]
print("Rotas spoofed (95ᵒ perc.):")
print(spoofed["route"].tolist())


