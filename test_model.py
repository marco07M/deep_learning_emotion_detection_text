from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import joblib


# Modell & Tokenizer laden
model_path = "./model_output"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# Label-Encoder laden
le = joblib.load(f"{model_path}/label_encoder.joblib")

# Falls GPU verfügbar → auf GPU verschieben
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).cpu().numpy()
        label = le.inverse_transform(prediction)[0]
    return label

print(predict("I love this movie!"))        # → Erwartet: 'joy' oder 'love'
print(predict("I'm so angry right now...")) # → Erwartet: 'anger'
print(predict("This is a neutral statement.")) # → Erwartet: 'neutral'
print(predict("I do not like this at all.")) # → Erwartet: 'sadness' oder 'fear'
print(predict("This is a great day!"))     # → Erwartet: 'joy' oder 'love'
print(predict("I hate this!"))   # → Erwartet: 'anger' oder 'sadness'
print(predict("I am not sure about this.")) # → Erwartet: 'neutral'
print(predict("I am very happy!")) # → Erwartet: 'joy' oder 'love'
print(predict("I am feeling sad.")) # → Erwartet: 'sadness'
print(predict("I am scared.")) # → Erwartet: 'fear'
print(predict("I never though it will be like this!")) # → Erwartet: 'joy' oder 'love'
print(predict("I am surprised.")) # → Erwartet: 'sadness'