import joblib

# 1. Modell laden
model = joblib.load('1_model.pkl')

# 2. Texte, die du klassifizieren möchtest
neue_texte = [
    "This product is amazing! I love it.",
    "im feeling quite sad and sorry for myself but ill snap out of it soon",
    "This is the worst experience I've ever had.",
    "I hate this product, it doesn't work at all.",
]

# 3. Vorhersagen
vorhersagen = model.predict(neue_texte)
wahrscheinlichkeiten = model.predict_proba(neue_texte)

for text, label, probs in zip(neue_texte, vorhersagen, wahrscheinlichkeiten):
    print(f"Text: {text!r}")
    print(f"→ Vorhergesagte Klasse: {label}")
    print(f"→ Klassen-Wahrscheinlichkeiten: {dict(zip(model.classes_, probs))}")
    print()
