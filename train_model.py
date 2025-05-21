#!/usr/bin/env python3
# train_transformer_classifier.py

import os
import sys
import pandas as pd
import numpy as np
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
import  torch

def main(
    data_path: str = "all.txt",
    model_name: str = "roberta-base",
    output_dir: str = "./model_output",
    test_size: float = 0.2,
    seed: int = 42,
    num_epochs: int = 3,
    train_batch_size: int = 8,
    eval_batch_size: int = 8,
    learning_rate: float = 5e-5,
):
    # 1. Lade deine Daten
    df = pd.read_csv(
        data_path,
        sep=";",
        header=None,
        names=["text", "label"],
        encoding="utf-8",
    ).dropna()

    # 2. Label-Encoding
    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["label"])
    num_labels = len(le.classes_)
    print(f"→ Gefundene Klassen ({num_labels}): {list(le.classes_)}")

    # 3. Train/Test-Split
    train_df, eval_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label_id"]
    )

    # 4. Erzeuge HuggingFace-Datasets
    ds_train = Dataset.from_pandas(train_df[["text", "label_id"]])
    ds_eval  = Dataset.from_pandas(eval_df[["text", "label_id"]])
    datasets = DatasetDict({"train": ds_train, "eval": ds_eval})

    # 5. Tokenizer und Model laden
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    # 6. Tokenisierung
    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True, padding=False)
    tokenized = datasets.map(tokenize_batch, batched=True)
    tokenized = tokenized.remove_columns(["text"])
    tokenized.set_format("torch")

    # 7. Data Collator für dynamisches Padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 8. Trainingsparameter
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

    # 9. Trainer initialisieren
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["eval"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 10. Training und Evaluation
    trainer.train()
    metrics = trainer.evaluate()
    print("Evaluationsergebnisse:", metrics)

    # 11. Speichere Modell, Tokenizer und Label-Encoder
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    # Label-Encoder sichern
    import joblib
    joblib.dump(le, os.path.join(output_dir, "label_encoder.joblib"))
    print(f"Modell, Tokenizer und Label-Encoder gespeichert in: {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_transformer_classifier.py path/to/train.txt [output_dir]")
        sys.exit(1)
    data_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "./model_output"
    main(data_path, output_dir=out_dir)
