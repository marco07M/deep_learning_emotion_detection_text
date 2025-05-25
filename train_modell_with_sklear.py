#!/usr/bin/env python3
# train_and_save_sgd_model.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
import joblib

# Pfad zu deinen Daten und Speicherort
DATA_PATH = 'data/all_balanced.txt'  # ggf. anpassen
MODEL_DIR = 'test2'
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Daten laden
df = pd.read_csv(
    DATA_PATH,
    sep=';',
    header=None,
    names=['text', 'label'],
    encoding='utf-8'
).dropna()

# 2. Train/Test-Split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# 3. Vektorisierung
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# 4. Inkrementeller Klassifikator (Logistische Regression via SGD)
clf = SGDClassifier(loss='log_loss', random_state=42)

# 5. Training über mehrere Epochen
EPOCHS = 100
train_acc, test_acc = [], []
classes = np.unique(y_train)

for epoch in range(1, EPOCHS + 1):
    clf.partial_fit(X_train_vec, y_train, classes=classes)
    train_score = clf.score(X_train_vec, y_train)
    test_score  = clf.score(X_test_vec, y_test)
    train_acc.append(train_score)
    test_acc.append(test_score)
    print(f"Epoche {epoch}/{EPOCHS} — Train: {train_score:.3f}, Test: {test_score:.3f}")

# 6. Grafik erstellen
plt.figure()
plt.plot(range(1, EPOCHS + 1), train_acc, label='Train accuracy')
plt.plot(range(1, EPOCHS + 1), test_acc,  label='Test accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Genauigkeit pro Epoche (SGDClassifier)')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# 7. Modell und Vektorisierer speichern
joblib.dump(clf, os.path.join(MODEL_DIR, 'sgd_classifier.joblib'))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, 'vectorizer.joblib'))
print(f"💾 Modell und Vektorisierer gespeichert in: {MODEL_DIR}")
