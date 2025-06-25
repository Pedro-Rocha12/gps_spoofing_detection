#!/usr/bin/env python3
import os
import glob
import shutil
from sklearn.model_selection import train_test_split

# diretórios de entrada
TRACKS_DIR    = "tracks"
SPOOFED_DIR   = "spoofed_tracks"

# diretórios de saída
OUT_ROOT      = "split"
OUT_TRAIN_DIR = os.path.join(OUT_ROOT, "train")
OUT_VAL_DIR   = os.path.join(OUT_ROOT, "val")
OUT_TEST_DIR  = os.path.join(OUT_ROOT, "test")

# cria todos os diretórios de output
for d in (OUT_TRAIN_DIR, OUT_VAL_DIR, OUT_TEST_DIR):
    os.makedirs(d, exist_ok=True)

# 1) lista TODOS os CSVs “normais” em tracks/
normal_paths = glob.glob(os.path.join(TRACKS_DIR, "*.csv"))

# 2) lista TODOS os CSVs “spoofed” em spoofed_tracks/
spoofed_paths = glob.glob(os.path.join(SPOOFED_DIR, "*.csv"))

# 3) divide apenas as rotas “normais” em train/val/test (80/10/10)
train_val_norm, test_norm = train_test_split(
    normal_paths, test_size=0.10, random_state=42
)
train_norm, val_norm = train_test_split(
    train_val_norm,
    test_size=0.1111,  # 0.1111 * 0.9 ≈ 0.10 do total
    random_state=42
)

# 4) copia os “normais”
for src in train_norm:
    shutil.copy(src, OUT_TRAIN_DIR)
for src in val_norm:
    shutil.copy(src, OUT_VAL_DIR)
for src in test_norm:
    shutil.copy(src, OUT_TEST_DIR)

# 5) copia TODOS os “spoofed” diretamente para test/
for src in spoofed_paths:
    shutil.copy(src, OUT_TEST_DIR)

# 6) relatório
print("✔️ Separação concluída:")
print(f"   • Normal total em '{TRACKS_DIR}': {len(normal_paths)} rotas")
print(f"     – train: {len(train_norm)}")
print(f"     –   val: {len(val_norm)}")
print(f"     –  test normais: {len(test_norm)}")
print(f"   • Spoofed total em '{SPOOFED_DIR}': {len(spoofed_paths)} rotas → todas em split/test/")
