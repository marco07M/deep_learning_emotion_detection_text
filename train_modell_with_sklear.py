#!/usr/bin/env python3
# test_train_save.py

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib   # <-- zum Speichern/Laden

filepath="data/all.txt"
model_path="model.pkl"
# 1. Daten einlesen
df = pd.read_csv(filepath, sep=';', header=None, names=['text','label'], encoding='utf-8')
df = df.dropna()

# 2. Prüfen, dass mindestens 2 Klassen da sind
if df['label'].nunique() < 2:
    print("Zu wenige Klassen zum Trainieren.")
    sys.exit(1)

# 3. Train/Test-Split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# 4. Pipeline aufsetzen
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', LogisticRegression(max_iter=1000)),
])

# 5. Trainieren
pipeline.fit(X_train, y_train)
acc = pipeline.score(X_test, y_test)
print(f"✅ Modell trainiert. Test-Accuracy: {acc:.2f}")

# 6. Modell speichern
joblib.dump(pipeline, model_path)
print(f"💾 Modell gespeichert in: {model_path}")

