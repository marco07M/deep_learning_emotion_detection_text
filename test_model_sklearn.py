import joblib

# 1. Modell und Vektorisierer laden
MODEL_DIR = "model"  # Pfad zum Verzeichnis, in dem sgd_classifier.joblib und vectorizer.joblib liegen
vectorizer = joblib.load(f"{MODEL_DIR}/vectorizer.joblib")
clf        = joblib.load(f"{MODEL_DIR}/sgd_classifier.joblib")

# 2. Neue Texte, die du klassifizieren willst
neue_texte = [
    "i almost feel funny not adding a picture at the bottom of my post like denis and dave",
    "I'm so angry right now...",
    "This is a neutral statement.",
    "I do not like this at all.",
    "This is a great day!",
    "I hate this!",
    "I am not sure about this.",
    "I am very happy!",
    "I am feeling sad.",
    "I am scared.",
    "I never thought it will be like this!",
    "I am surprised."
]

# 3. Texte vektorisieren
X_new = vectorizer.transform(neue_texte)

# 4. Vorhersagen
predictions = clf.predict(X_new)
probabilities = clf.predict_proba(X_new)  # Falls du auch Wahrscheinlichkeiten brauchst

# 5. Ergebnisse ausgeben
for text, label, probs in zip(neue_texte, predictions, probabilities):
    print(f"Text: {text!r}")
    print(f"→ Vorhergesagte Klasse: {label}")
    # Klassen-Reihenfolge findest du in clf.classes_
    print(f"→ Wahrscheinlichkeiten: {dict(zip(clf.classes_, probs))}\n")
