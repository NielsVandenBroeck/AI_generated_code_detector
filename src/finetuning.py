import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback, BertTokenizer, BertForSequenceClassification
import numpy as np
import evaluate
from datasets import load_dataset, concatenate_datasets
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report
import pandas as pd


class FinetuningPipeline:
    def __init__(self, model_name, dataset_config, evaluate_dataset_names, epochs):
        self.model_name = model_name
        self.dataset_config = dataset_config
        self.evaluate_dataset_names = evaluate_dataset_names

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.metric = evaluate.load("accuracy")

        self.trainer = None
        self.training_args = TrainingArguments(
            output_dir="../training",
            eval_strategy="no",
            logging_dir="../training/logs",
            logging_steps=100,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=epochs
        )

    def setup_trainer(self, train_dataset=None, eval_dataset=None):
        if train_dataset is None or eval_dataset is None:
            train_dataset, eval_dataset = self.load_and_prepare_datasets()

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics
        )

    def load_and_prepare_datasets(self):
        combined_datasets = []
        for name, num_samples in self.dataset_config.items():
            dataset_path = f"../training/labeled_dataset_{name}.jsonl"
            dataset = load_dataset("json", data_files=dataset_path, split="train")
            dataset = dataset.shuffle(seed=42).select(range(num_samples))
            combined_datasets.append(dataset)

        dataset = concatenate_datasets(combined_datasets).shuffle(seed=42)

        train_valid = dataset.train_test_split(test_size=0.2, seed=42)
        tokenized_train = train_valid["train"].map(self.tokenize_function, batched=True)
        tokenized_valid = train_valid["test"].map(self.tokenize_function, batched=True)

        return tokenized_train.select(range(2000)), tokenized_valid.select(range(400))

    def flip_labels(self, example):
        example["label"] = 1 - example["label"]
        return example

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        cm = confusion_matrix(labels, predictions, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', zero_division=0
        )
        acc = self.metric.compute(predictions=predictions, references=labels)["accuracy"]

        tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) != 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) != 0 else 0

        print(classification_report(labels, predictions, digits=4))

        return {
            "ACC": acc,
            "F1": f1,
            "Precision": precision,
            "TPR": tpr,
            "TNR": tnr,
            "FPR": fpr,
            "FNR": fnr,
        }

    def train_model(self):
        train_dataset, eval_dataset = self.load_and_prepare_datasets()

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics
        )

        self.trainer.train()

    def save_model(self, output_model_name):
        self.model.save_pretrained(f"../models/{output_model_name}")
        self.tokenizer.save_pretrained(f"../models/{output_model_name}")

    def evaluate_all_datasets(self):
        all_metrics = []

        for test_key, test_name in self.evaluate_dataset_names.items():
            print(f"\n=== Evaluating on dataset: {test_key}:{test_name} ===")
            test_data = load_dataset("json", data_files={"test": f"../training/labeled_dataset_{test_key}.jsonl"})

            tokenized = test_data["test"].map(self.tokenize_function, batched=True)
            test_dataset = tokenized.shuffle(seed=42).select(range(1000))

            pred_output = self.trainer.predict(test_dataset)
            logits, labels = pred_output.predictions, pred_output.label_ids
            metrics = self.compute_metrics((logits, labels))
            metrics["Loss"] = pred_output.metrics["test_loss"]
            metrics["Dataset"] = test_name
            all_metrics.append(metrics)


def main():
    dataset_config = {
        "baseline": 1200,
        "no_comments": 1000,
        "replace_identifiers": 600,
        "no_comments_replace_identifiers": 400,
        "complete": 200,
        "no_comments_replace_identifiers_complete": 200,
        "formatted": 200,
        "no_comments_replace_identifiers_formatted": 200,
    }

    evaluate_dataset_names = {
        "baseline": "baseline",
        "correct": "passed code",
        "no_comments": "removed comments (RC)",
        "replace_identifiers": "replaced identifiers (RI)",
        "formatted": "formatted code",
        "complete": "complete code",
        "prompt_no_comments": "prompt no comments",
        "mimic_person": "mimic person",
        "o3mini": "o3mini",
        "4o": "4o",
        "no_comments_replace_identifiers": "RC + RI",
        "no_comments_replace_identifiers_correct": "RC + RI + passed code",
        "no_comments_replace_identifiers_formatted": "RC + RI + formatted code",
        "no_comments_replace_identifiers_complete": "RC + RI + complete code",
        "no_comments_replace_identifiers_mimic_person": "RC + RI + mimic person",
        "no_comments_replace_identifiers_model_o3mini": "RC + RI + o3mini",
        "no_comments_replace_identifiers_model_4o": "RC + RI + 4o",
        "temperature_0.0": "temperature 0.0",
        "temperature_0.3": "temperature 0.3",
        "temperature_0.7": "temperature 0.7",
        "temperature_1.0": "temperature 1.0",
        "temperature_1.3": "temperature 1.3",
        "temperature_1.7": "temperature 1.7",
        "temperature_2.0": "temperature 2.0"
    }

    model_name = "roberta-base"
    pipeline = FinetuningPipeline(model_name, dataset_config, evaluate_dataset_names, 1)
    pipeline.setup_trainer()

    pipeline.train_model()
    pipeline.save_model("../models/testest")
    #pipeline.evaluate_all_datasets()


if __name__ == "__main__":
    main()
