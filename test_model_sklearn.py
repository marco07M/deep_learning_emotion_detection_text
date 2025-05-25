import joblib

# 1. Modell laden
model = joblib.load('path/to/model.joblib')

# 2. Texte, die du klassifizieren möchtest
neue_texte = [
    "Das Produkt war hervorragend und ich bin sehr zufrieden.",
    "Leider hat der Service gar nicht mit meinen Erwartungen übereingestimmt."
]

# 3. Vorhersagen
vorhersagen = model.predict(neue_texte)
wahrscheinlichkeiten = model.predict_proba(neue_texte)

for text, label, probs in zip(neue_texte, vorhersagen, wahrscheinlichkeiten):
    print(f"Text: {text!r}")
    print(f"→ Vorhergesagte Klasse: {label}")
    print(f"→ Klassen-Wahrscheinlichkeiten: {dict(zip(model.classes_, probs))}")
    print()
