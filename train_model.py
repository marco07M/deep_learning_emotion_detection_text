import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import joblib

# --- GPU prüfen ---
print("CUDA verfügbar:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Verwendete GPU:", torch.cuda.get_device_name(0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Parameter ---
data_path = "data/all.txt"
model_name = "roberta-base"
output_dir = "./model_output"
test_size = 0.2
seed = 42
num_epochs = 1
train_batch_size = 8
eval_batch_size = 8
learning_rate = 5e-5

# --- 1. Daten laden ---
df = pd.read_csv(data_path, sep=";", header=None, names=["text", "label"], encoding="utf-8").dropna()

# --- 2. Label-Encoding ---
le = LabelEncoder()
df["label_id"] = le.fit_transform(df["label"])
num_labels = len(le.classes_)
print(f"→ Gefundene Klassen ({num_labels}): {list(le.classes_)}")

# --- 3. Train/Test-Split ---
train_df, eval_df = train_test_split(
    df, test_size=test_size, random_state=seed, stratify=df["label_id"]
)

# --- 4. HuggingFace-Datasets erzeugen ---
ds_train = Dataset.from_pandas(train_df[["text", "label_id"]].rename(columns={"label_id": "labels"}))
ds_eval  = Dataset.from_pandas(eval_df[["text", "label_id"]].rename(columns={"label_id": "labels"}))
datasets = DatasetDict({"train": ds_train, "eval": ds_eval})

# --- 5. Tokenizer & Modell laden ---
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
model.to(device)  # Modell auf GPU verschieben

# --- 6. Tokenisierung ---
def tokenize_batch(batch):
    return tokenizer(batch["text"], truncation=True, padding=False)

tokenized = datasets.map(tokenize_batch, batched=True)
tokenized = tokenized.remove_columns(["text"])
tokenized.set_format("torch")

# --- 7. Data Collator ---
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# --- 8. Trainingsargumente ---
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,
    learning_rate=learning_rate,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=2,
    seed=seed,
)

# --- 9. Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["eval"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# --- 10. Training + Evaluation ---
trainer.train()
metrics = trainer.evaluate()
print("Evaluationsergebnisse:", metrics)

# --- 11. Modell speichern ---
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
joblib.dump(le, os.path.join(output_dir, "label_encoder.joblib"))
print(f"Modell, Tokenizer und Label-Encoder gespeichert in: {output_dir}")
