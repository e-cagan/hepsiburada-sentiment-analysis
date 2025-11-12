"""
Görselleştirmeleri içeren modül.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay
import joblib

# Config
ART = Path("artifacts/baseline")
DATA_PATH = Path("data/hepsiburada_balanced_300k.csv")
OUT_DIR = Path("reports/baseline_plots"); OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1) Sınıf dağılımı
df = pd.read_csv(DATA_PATH)
ax = df['label'].value_counts().plot(kind="bar")
plt.title("Label Distribution")
plt.tight_layout()
plt.savefig(OUT_DIR / "label_distribution.png")
plt.close()

# 2) Metin uzunluğu dağılımı
df['len'] = df['combined_text'].astype(str).str.split().apply(len)
df['len'].plot(kind="hist", bins=50)
plt.title("Text Length Histogram")
plt.tight_layout()
plt.savefig(OUT_DIR / "length_hist.png")
plt.close()

# 3) Confusion matrix (best model)
best = (ART / "best_model.txt").read_text().strip()
cm = json.loads((ART / f"{best}_confusion_matrix.json").read_text())
cm = np.array(cm)

labels = sorted(df['label'].astype(str).unique())  # basit etiket seti
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(xticks_rotation=45)
plt.title(f"Confusion Matrix - {best}")
plt.tight_layout()
plt.savefig(OUT_DIR / f"cm_{best}.png")
plt.close()
