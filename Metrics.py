#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate_metrics_from_existing_model.py

Dieses Skript lädt:
  - den balancierten Datensatz (data/all_balanced.txt),
  - das gespeicherte SGDClassifier-Modell (model/sgd_classifier.joblib) und
  - den gespeicherten CountVectorizer (model/vectorizer.joblib)

Anschließend wird auf dem Testset (20 % stratified split) folgendes berechnet und ausgegeben:
  1. Gesamt-Accuracy
  2. Precision / Recall / F1-Score pro Klasse
  3. Makro- und gewichtete Durchschnittswerte
  4. Konfusionsmatrix (Rohdaten + Heatmap)

Voraussetzungen:
  - Python 3.x
  - pandas, numpy, scikit-learn, matplotlib, joblib installiert
  - Das Verzeichnis `model/` enthält `sgd_classifier.joblib` und `vectorizer.joblib`
  - Das Verzeichnis `data/` enthält `all_balanced.txt`
"""

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

# ----------------------------------------------------------------
# 1. Pfade konfigurieren
# ----------------------------------------------------------------

# Pfad zu den balancierten Rohdaten (Semikolon-Separator: text;label)
DATA_PATH = os.path.join("data", "all_balanced.txt")

# Verzeichnis, in dem das Modell und der Vektorisierer gespeichert sind
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "sgd_classifier.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")

# ----------------------------------------------------------------
# 2. Daten einlesen und Train/Test-Split
# ----------------------------------------------------------------

# 2.1 Daten laden
df = pd.read_csv(
    DATA_PATH,
    sep=";",
    header=None,
    names=["text", "label"],
    encoding="utf-8"
).dropna()

# 2.2 Stratified Train/Test-Split (80% / 20%) mit random_state=42
X = df["text"].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Klassenliste sortieren (alphabetisch), um Konsistenz in Metriken und Konfusionsmatrix zu gewährleisten
classes = np.unique(y_train)
classes_sorted = np.sort(classes)

print("\n=== Datensatz-Informationen ===")
print(f"Gesamtzahl Beispiele:   {len(df)}")
print(f"Trainingsset-Größe:     {len(X_train)}")
print(f"Testset-Größe:          {len(X_test)}")
print(f"Emotionen (Klassen):    {list(classes_sorted)}")
print("=" * 30)

# ----------------------------------------------------------------
# 3. Modell und Vektorisierer laden
# ----------------------------------------------------------------

print("\nLade gespeichertes Modell und Vektorisierer…")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modell-Datei nicht gefunden: {MODEL_PATH}")
if not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError(f"Vektorisierer-Datei nicht gefunden: {VECTORIZER_PATH}")

clf = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# ----------------------------------------------------------------
# 4. Testdaten transformieren & Vorhersagen erstellen
# ----------------------------------------------------------------

print("Transformiere Testdaten mit CountVectorizer…")
X_test_vec = vectorizer.transform(X_test)

print("Erstelle Vorhersagen auf dem Testset…")
y_pred = clf.predict(X_test_vec)

# ----------------------------------------------------------------
# 5. Metriken berechnen und ausgeben
# ----------------------------------------------------------------

print("\n=== Leistungsmetriken ===")

# 5.1 Gesamt-Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"\n1) Gesamt-Accuracy: {acc:.4f}")

# 5.2 Precision / Recall / F1-Score pro Klasse
print("\n2) Precision / Recall / F1-Score pro Klasse und Support:")
report_dict = classification_report(
    y_test,
    y_pred,
    target_names=classes_sorted,
    digits=4,
    output_dict=True
)
report_df = pd.DataFrame(report_dict).transpose()
# Runde die Werte für bessere Lesbarkeit
report_df[["precision", "recall", "f1-score", "support"]] = \
    report_df[["precision", "recall", "f1-score", "support"]].round(4)
print(report_df, "\n")

# 5.3 Makro- und gewichteter Durchschnitt (F1-Score)
if "macro avg" in report_dict and "weighted avg" in report_dict:
    macro_f1 = report_dict["macro avg"]["f1-score"]
    weighted_f1 = report_dict["weighted avg"]["f1-score"]
else:
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
print(f"3) Makro-Avg F1-Score:      {macro_f1:.4f}")
print(f"   Gewichtetes Avg F1-Score: {weighted_f1:.4f}")

# 5.4 Konfusionsmatrix (Rohdaten)
cm = confusion_matrix(y_test, y_pred, labels=classes_sorted)
print("\n4) Konfusionsmatrix (Zeilen = wahre Labels, Spalten = vorhergesagte Labels):")
print(cm)

# ----------------------------------------------------------------
# 6. Konfusionsmatrix als Heatmap mit Matplotlib plotten
# ----------------------------------------------------------------

print("\nErstelle Heatmap der Konfusionsmatrix…")
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

# Achsenbeschriftungen setzen
ax.set(
    xticks=np.arange(len(classes_sorted)),
    yticks=np.arange(len(classes_sorted)),
    xticklabels=classes_sorted,
    yticklabels=classes_sorted,
    ylabel="Wahre Emotion",
    xlabel="Vorhergesagte Emotion",
    title="Konfusionsmatrix"
)

# X-Tick-Labels rotieren für bessere Lesbarkeit
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Zellwerte in die Heatmap schreiben
fmt = "d"
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j,
            i,
            format(cm[i, j], fmt),
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black"
        )

fig.tight_layout()
plt.show()

# ----------------------------------------------------------------
# 7. (Optional) 5-Fold Kreuzvalidierung
# ----------------------------------------------------------------

# Wenn gewünscht, kann man hier einen 5-fachen CV-Block aktivieren.
# Standardmäßig ist er deaktiviert (False). Bei True wird eine Cross-Val auf dem gesamten balancierten Set durchgeführt.

do_crossval = False  # Setze auf True, um 5-Fold CV durchzuführen

if do_crossval:
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.linear_model import SGDClassifier as SGDClassifierCV

    print("\n=== 5-Fold Kreuzvalidierung (Macro-F1) ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_all_vec = vectorizer.transform(X)  # komplette Feature-Matrix
    y_all = y

    clf_cv = SGDClassifierCV(loss="log_loss", random_state=42)

    scores = cross_val_score(
        clf_cv,
        X_all_vec,
        y_all,
        cv=skf,
        scoring="f1_macro",
        n_jobs=-1
    )
    print(f"5-Fold Macro-F1: {scores.mean():.4f} ± {scores.std():.4f}")

# ----------------------------------------------------------------
# Ende des Skripts
# ----------------------------------------------------------------
