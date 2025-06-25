# src/prepare_val_labels_with_spoof.py
import glob, os
import numpy as np

# 1) defina o set com os 41 nomes
spoofed = {
    "statevectors_e49444_1711296244_1711319996.csv",
    "statevectors_e48c04_1711562172_1711578294.csv",
    "statevectors_e49406_1711389103_1711405667.csv",
    "statevectors_e48db0_1711648555_1711665305.csv",
    "statevectors_e48eff_1711044184_1711060059.csv",
    "statevectors_e4952f_1711475800_1711492351.csv",
    "statevectors_e49b52_1711132188_1711148144.csv",
    "statevectors_e486c3_1711300201_1711313169.csv",
    "statevectors_e48043_1711129059_1711142125.csv",
    "statevectors_e4827e_1711042927_1711055700.csv",
    "statevectors_e48ad4_1711472703_1711486508.csv",
    
}

# 2) Encontre todos os CSVs que estão no split de validação
val_dir = "data/split_output/val"
paths = sorted(glob.glob(os.path.join(val_dir, "statevectors_*.csv")))

# 3) Para cada arquivo, atribua 1 se estiver em spoofed, senão 0
labels = []
for p in paths:
    fname = os.path.basename(p)
    labels.append(1 if fname in spoofed else 0)

labels = np.array(labels, dtype=int)
np.savez(os.path.join(val_dir, "val_labels_with_spoof.npz"), y=labels)

print(f"✔️ y_val com spoofing salvo em {val_dir}/val_labels_with_spoof.npz "
      f"— {labels.sum()} positivos, {len(labels)-labels.sum()} negativos")
