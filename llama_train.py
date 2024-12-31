#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ------------------------------
# 1) Setup: Device + Model Paths
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_model_dir = './best_llama_model'

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(best_model_dir)
model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
model.to(device)
model.eval()

# If needed, set pad_token to eos_token (e.g., for LLaMA-based models)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# ------------------------------
# 2) Helper Functions
# ------------------------------
def predict_labels(texts, batch_size=4):
    """
    Given a list of text entries, return:
      - predicted labels (argmax of logits)
      - logits/probs
    """
    all_preds = []
    all_logits = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        # Tokenize
        encodings = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # shape: (batch_size, num_labels)

        preds = torch.argmax(logits, dim=1)  # class indices
        all_preds.extend(preds.cpu().numpy())
        all_logits.extend(logits.cpu().numpy())

    return np.array(all_preds), np.array(all_logits)

def logits_to_risk_score(logits, min_score=0, max_score=10):
    """
    Optional: Map model logits to a 0–10 scale (heuristic).
    """
    max_logits = logits.max(axis=1)
    # Convert logit to a [0..1] “prob” using a logistic function (example)
    risk_probs = 1 / (1 + np.exp(-max_logits))
    # Map from [0..1] to [0..10]
    scaled_risk = risk_probs * (max_score - min_score) + min_score
    # Round to int
    scaled_risk_int = scaled_risk.round().astype(int)
    return scaled_risk_int

# ------------------------------
# 3) Inference Over a Directory
# ------------------------------
def process_csv_directory(input_dir, output_dir, file_suffix="_with_predictions.csv"):
    """
    Loop over all CSV files in 'input_dir'. For each CSV:
      1) Read into DF
      2) Predict labels + (optional) risk scores
      3) Save output to 'output_dir'
    """
    # Create the output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Get all CSV files in input_dir
    all_csv_files = [
        f for f in os.listdir(input_dir)
        if f.endswith(".csv")
    ]

    for csv_file in all_csv_files:
        input_path = os.path.join(input_dir, csv_file)
        print(f"Processing {input_path}...")

        # 1) Read CSV
        df = pd.read_csv(input_path)
        if "text" not in df.columns:
            print(f"Skipping {csv_file}: 'text' column not found.")
            continue

        # 2) Predict
        predicted_labels, logits = predict_labels(df["text"].tolist())
        risk_scores = logits_to_risk_score(logits)

        # 3) Attach to DF + save
        df["predicted_label"] = predicted_labels
        df["risk_score_0_10"] = risk_scores

        # Build output filename
        base_name = os.path.splitext(csv_file)[0]
        output_filename = f"{base_name}{file_suffix}"
        output_path = os.path.join(output_dir, output_filename)

        df.to_csv(output_path, index=False)
        print(f"Saved results to {output_path}")

# ------------------------------
# 4) Main Execution
# ------------------------------
if __name__ == "__main__":
    # Example usage:
    input_directory = "./processed_data"
    output_directory = "./processed_data_with_predictions"

    process_csv_directory(input_directory, output_directory)
    print("Done processing all CSV files!")
