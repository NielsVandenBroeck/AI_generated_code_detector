import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import torch

model_path = "../models/test_model_combinations_25epochs"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

def get_prediction(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length")
    logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    pred_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_class].item()
    return pred_class, confidence

def line_importance_explanation(code_sample, max_block_size=15):
    lines = code_sample.strip().split('\n')
    original_pred, original_conf = get_prediction(code_sample)

    print(f"Original AI score: {original_conf:.4f}\n")

    results = []

    for block_size in range(1, max_block_size + 1):
        for i in range(len(lines) - block_size + 1):
            # Remove a block of `block_size` lines starting at line `i`
            removed_block = lines[i:i + block_size]
            modified_lines = lines[:i] + lines[i + block_size:]
            modified_code = '\n'.join(modified_lines)
            #print(f"\n\n Mofidied:\n{modified_code}")
            mod_pred, mod_conf = get_prediction(modified_code)
            confidence_change = mod_conf - original_conf
            results.append(((i + 1, i + block_size), removed_block, confidence_change))

            print(f"Removed lines {i + 1}-{i + block_size}: Pred = {mod_pred}, Conf = {mod_conf:.4f}, "
                  f"ΔConf = {confidence_change:.4f}, Flip = {original_pred != mod_pred}")

    # Sort by absolute importance (highest first)
    results.sort(key=lambda x: abs(x[2]), reverse=True)
    return results

def extract_comments(code):
    lines = code.split('\n')
    comments = []
    for line in lines:
        match = re.search(r'#.*', line)
        if match:
            comments.append(match.group())
    return comments

def insert_comments(human_written_sample, ai_generated_sample):
    human_lines = human_written_sample.split('\n')
    comments = extract_comments(ai_generated_sample)
    original_pred, original_conf = get_prediction(human_written_sample)

    print(f"Original human sample prediction: {original_pred} ({original_conf:.4f})\n")

    for comment in comments:
        print(f"Trying comment: {comment}\n")
        mean_conf = 0
        for i in range(len(human_lines) + 1):  # +1 to allow insertion at the end
            modified_lines = human_lines[:i] + [comment] + human_lines[i:]
            modified_code = '\n'.join(modified_lines)
            mod_pred, mod_conf = get_prediction(modified_code)
            confidence_change = abs(mod_conf - original_conf) if mod_pred == original_pred else abs(original_conf - (1 - mod_conf))
            mean_conf += confidence_change
            print(f"Inserted at line {i + 1}: Pred = {mod_pred}, Conf = {mod_conf:.4f}, "
                  f"ΔConf = {confidence_change:.4f}, Flip = {mod_pred != original_pred}")
        print(f"confidence change: {mean_conf/9}")
        print('-' * 60 + '\n')
    return

def normalize_code(code: str) -> str:
    code = code.replace('\r\n', '\n').replace('\r', '\n')
    lines = [line.rstrip() for line in code.strip().split('\n')]
    return '\n'.join(lines)

def main():
    df_comments = pd.read_csv("../datasets/baseline.csv")

    ai_generated_sample = df_comments["ai_generated_code"][875]
    human_written_sample = normalize_code(df_comments["code"][875])
    print(ai_generated_sample)
    insert_comments(human_written_sample, ai_generated_sample)

    ai_generated_sample = df_comments["ai_generated_code"][18066]
    print(ai_generated_sample)
    line_importance_explanation(ai_generated_sample)


if __name__ == "__main__":
    main()
