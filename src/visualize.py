# src/visualize.py
import matplotlib.pyplot as plt
import pandas as pd

def plot_anomalies(df, score_col, threshold=None):
    """Plota os scores ao longo do tempo e destaca anomalias."""
    df = df.sort_values('datetime')
    plt.figure(figsize=(12,4))
    plt.plot(df['datetime'], df[score_col], label='score')
    if threshold is not None:
        plt.axhline(threshold, color='r', linestyle='--', label='threshold')
    plt.legend(); plt.title('Anomaly score ao longo do tempo')
    plt.show()
