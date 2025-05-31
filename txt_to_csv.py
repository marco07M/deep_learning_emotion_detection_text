import csv

# Pfad zur Eingabedatei (Textdatei)
input_file = 'data/all_balanced.txt'
# Pfad zur Ausgabedatei (CSV-Datei)
output_file = 'data/all_balanced.csv'

# Öffne die Eingabedatei und die Ausgabedatei
with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', newline='', encoding='utf-8') as outfile:

    writer = csv.writer(outfile)
    # Schreibe die Kopfzeile
    writer.writerow(['text', 'label'])

    for line in infile:
        # Entferne mögliche Leerzeichen oder Zeilenumbrüche
        line = line.strip()
        if not line:
            continue  # Überspringe leere Zeilen
        try:
            text, label = line.split(';')
            writer.writerow([text.strip(), label.strip()])
        except ValueError:
            print(f"Warnung: Zeile konnte nicht verarbeitet werden: {line}")
