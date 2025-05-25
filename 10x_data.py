import pandas as pd
from sklearn.utils import resample

# --- Pfade anpassen ---
INPUT_PATH = 'data/all.txt'
OUTPUT_PATH = 'data/all_balanced.txt'

# 1. Daten laden
df = pd.read_csv(INPUT_PATH, sep=';', header=None, names=['text', 'label'], encoding='utf-8').dropna()

# 2. Ursprüngliche Häufigkeiten anzeigen
print("Original counts per label:")
print(df['label'].value_counts())

# 3. Zielhöhe = maximaler Klassenumfang
max_count = df['label'].value_counts().max()

# 4. Oversampling jeder Klasse auf max_count
balanced_frames = []
for label, group in df.groupby('label'):
    # Gruppe belassen
    balanced_frames.append(group)
    # Fehlende Anzahl ergänzen (mit Replacement)
    if len(group) < max_count:
        resampled = resample(
            group,
            replace=True,
            n_samples=max_count - len(group),
            random_state=42
        )
        balanced_frames.append(resampled)

# 5. Zusammenführen und mischen
balanced_df = pd.concat(balanced_frames).sample(frac=1, random_state=42).reset_index(drop=True)

# 6. Balancierte Häufigkeiten anzeigen
print("\nBalanced counts per label:")
print(balanced_df['label'].value_counts())

# 7. In neue Datei speichern
balanced_df.to_csv(OUTPUT_PATH, sep=';', header=False, index=False, encoding='utf-8')
print(f"\n💾 Balancierte Datei gespeichert unter: {OUTPUT_PATH}")
