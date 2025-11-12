"""
Sentiment analysis pipelıne'ını içeren modül.
"""

import re
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Config
DATA_PATH = Path("data/hepsiburada_balanced_300k.csv")
OUT_DIR = Path("artifacts/baseline")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

# Basit TR odaklı temizlik
TR_STOPWORDS = set("""
acaba ama aslında az bazı belki bile çünkü daha de da değil eğer fakat gibi hem ki
ile ise için kadar nasıl ne neden değil mi mu mü mı mı̈ öyle ancak sonra çok
ve veya ya yani şu bu o şunlar bunlar onlar
""".split())

def basic_clean(text):
    """
    Metin temizleme fonksiyonu. (URL, hashtag, noktalama işaretleri ve sayıları temizler.)
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)              # URL
    text = re.sub(r"@\w+|#\w+", " ", text)                     # mention/hashtag
    text = re.sub(r"[^\w\sçğıöşü]", " ", text)                 # noktalama
    text = re.sub(r"\d+", " ", text)                           # sayılar
    tokens = [t for t in text.split() if t not in TR_STOPWORDS and len(t) > 2]
    return " ".join(tokens)

# Veriyi oku
df = pd.read_csv(DATA_PATH)

# Zorunlu kolon guard
assert 'combined_text' in df.columns, "CSV içinde 'combined_text' kolonu yok."
assert 'label' in df.columns, "CSV içinde 'label' (veya hedef) kolonu yok."

# NaN ve lowercase
df = df.dropna(subset=['combined_text', 'label']).copy()
df['combined_text'] = df['combined_text'].astype(str).str.lower()

# Temizlik kolonu (ayrı tutmak yararlı)
df['clean_text'] = df['combined_text'].apply(basic_clean)

# Train/Val split (label dengesini koru)
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'],
    test_size=0.2, random_state=RANDOM_STATE, stratify=df['label']
)

# ---- İki pipeline dene: TF-IDF + LR, TF-IDF + NB ----
pipelines = {
    "tfidf_logreg": make_pipeline(
        TfidfVectorizer(max_features=50000, ngram_range=(1,2), min_df=2),
        LogisticRegression(max_iter=200, n_jobs=-1, class_weight='balanced')
    ),
    "tfidf_nb": make_pipeline(
        TfidfVectorizer(max_features=50000, ngram_range=(1,2), min_df=2),
        MultinomialNB()
    ),
    "bow_logreg": make_pipeline(
        CountVectorizer(max_features=50000, ngram_range=(1,2), min_df=2),
        LogisticRegression(max_iter=200, n_jobs=-1, class_weight='balanced')
    ),
}

reports = {}
cms = {}

for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    reports[name] = rep
    cms[name] = cm.tolist()  # JSON friendly

    # Artefact olarak kayıt
    joblib.dump(pipe, OUT_DIR / f"{name}.joblib")
    
    with open(OUT_DIR / f"{name}_report.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    with open(OUT_DIR / f"{name}_confusion_matrix.json", "w", encoding="utf-8") as f:
        json.dump(cms[name], f, ensure_ascii=False, indent=2)

# En iyi modeli seç (macro F1 üzerinden)
def macro_f1(rep_dict): 
    return rep_dict['macro avg']['f1-score']

best_name = max(reports, key=lambda k: macro_f1(reports[k]))
with open(OUT_DIR / "best_model.txt", "w") as f:
    f.write(best_name + "\n")

print(f"Best model: {best_name} | macro F1 = {macro_f1(reports[best_name]):.4f}")

# Örnek inference kaydı
examples = [
    "ürün kesinlikle tavsiye etmiyorum, iade edeceğim",
    "kargo hızlıydı, satıcı çok ilgili, paketleme harika",
]
best_pipe = joblib.load(OUT_DIR / f"{best_name}.joblib")
preds = best_pipe.predict(examples)
for t, p in zip(examples, preds):
    print(">>>", t, " -> ", p)
