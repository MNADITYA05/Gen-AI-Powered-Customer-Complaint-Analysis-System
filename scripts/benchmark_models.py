"""
scripts/benchmark_models.py
----------------------------
State-of-the-art model comparison for complaint text classification.

Compares 6 models across 3 classification tasks (category, severity, emotion):
  1. TF-IDF + Naive Bayes          (classical baseline)
  2. TF-IDF + Logistic Regression  (current production model)
  3. TF-IDF + Linear SVM           (strongest classical text classifier)
  4. TF-IDF + Random Forest        (ensemble baseline)
  5. DistilBERT                    (lightweight transformer)
  6. FinBERT (ProsusAI/finbert)    (finance-domain BERT)

Metrics reported per model × task:
  - Accuracy
  - Weighted F1
  - Training time (seconds)
  - Inference time per sample (ms)

Usage:
    python3 scripts/benchmark_models.py \\
        --input  data/cfpb_prepared.csv \\
        --output data/benchmark_results.csv \\
        --sample 2000        # rows used (keep ≤2000 for transformer speed on CPU)
        --skip-transformers  # flag to skip DistilBERT/FinBERT if no time/GPU
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

TASKS = ["category", "severity", "emotion"]
RANDOM_STATE = 42


# ── Classical model definitions ───────────────────────────────────────────────

def build_classical_models() -> dict:
    tfidf = lambda: TfidfVectorizer(  # noqa: E731
        max_features=30_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    return {
        "Naive Bayes": Pipeline([
            ("tfidf", tfidf()),
            ("clf",   MultinomialNB()),
        ]),
        "Logistic Regression (current)": Pipeline([
            ("tfidf", tfidf()),
            ("clf",   LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)),
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", tfidf()),
            ("clf",   LinearSVC(max_iter=2000, C=1.0, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=10_000, sublinear_tf=True)),
            ("clf",   RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)),
        ]),
    }


# ── Transformer helpers ───────────────────────────────────────────────────────

def _transformer_available() -> bool:
    try:
        import torch          # noqa: F401
        import transformers   # noqa: F401
        return True
    except ImportError:
        return False


def _run_transformer(
    model_name: str,
    X_train: list[str],
    X_test:  list[str],
    y_train: list[str],
    y_test:  list[str],
    task: str,
) -> dict:
    """Fine-tune a HuggingFace sequence-classification model and evaluate."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"      [{model_name}] device={device}")

    label2id = {l: i for i, l in enumerate(sorted(set(y_train)))}
    id2label = {i: l for l, i in label2id.items()}
    num_labels = len(label2id)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    ).to(device)

    class TextDataset(Dataset):
        def __init__(self, texts, labels):
            self.enc = tokenizer(
                texts, padding=True, truncation=True,
                max_length=128, return_tensors="pt"
            )
            self.labels = torch.tensor([label2id[l] for l in labels])

        def __len__(self): return len(self.labels)

        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.enc.items()}, self.labels[idx]

    train_ds = TextDataset(X_train, y_train)
    test_ds  = TextDataset(X_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=32)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0,
        num_training_steps=len(train_loader) * 3,
    )

    # ── Train 3 epochs ──────────────────────────────────────────────────────
    t0 = time.time()
    model.train()
    for epoch in range(3):
        for batch, labels in train_loader:
            batch   = {k: v.to(device) for k, v in batch.items()}
            labels  = labels.to(device)
            outputs = model(**batch, labels=labels)
            outputs.loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    train_time = time.time() - t0

    # ── Evaluate ─────────────────────────────────────────────────────────────
    model.eval()
    preds_all = []
    t1 = time.time()
    with torch.no_grad():
        for batch, _ in test_loader:
            batch  = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds_all.extend(logits.argmax(dim=-1).cpu().tolist())
    infer_time = (time.time() - t1) / len(y_test) * 1000  # ms per sample

    y_pred = [id2label[p] for p in preds_all]
    return {
        "accuracy":    accuracy_score(y_test, y_pred),
        "f1":          f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "train_time":  train_time,
        "infer_ms":    infer_time,
    }


# ── Main benchmark loop ───────────────────────────────────────────────────────

def run_benchmark(input_path: str, output_path: str, sample: int, skip_transformers: bool) -> None:
    print(f"\n{'='*70}")
    print("  STATE-OF-THE-ART MODEL BENCHMARK")
    print(f"  Dataset: {input_path}   Sample: {sample}   Tasks: {TASKS}")
    print(f"{'='*70}\n")

    df = pd.read_csv(input_path)
    if sample and len(df) > sample:
        df = df.groupby("category", group_keys=False).apply(
            lambda g: g.sample(min(len(g), sample // df["category"].nunique()), random_state=RANDOM_STATE)
        ).reset_index(drop=True)
    print(f"Using {len(df):,} samples\n")

    results = []

    for task in TASKS:
        if task not in df.columns:
            print(f"Skipping '{task}' — column not found.\n")
            continue

        valid = df.dropna(subset=["complaint_text", task])
        valid = valid[valid[task].str.strip() != ""]
        X = valid["complaint_text"].tolist()
        y = valid[task].tolist()

        # Minimum 2 classes required
        if len(set(y)) < 2:
            print(f"Skipping '{task}' — only 1 class present.\n")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        print(f"── Task: {task.upper()} ({'|'.join(sorted(set(y)))}) ──")
        print(f"   Train: {len(X_train)}   Test: {len(X_test)}\n")

        # ── Classical models ──────────────────────────────────────────────────
        for model_name, pipeline in build_classical_models().items():
            print(f"   Training {model_name} …", end=" ", flush=True)
            t0 = time.time()
            pipeline.fit(X_train, y_train)
            train_time = time.time() - t0

            t1 = time.time()
            y_pred = pipeline.predict(X_test)
            infer_ms = (time.time() - t1) / len(X_test) * 1000

            acc = accuracy_score(y_test, y_pred)
            f1  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            print(f"acc={acc:.3f}  f1={f1:.3f}  train={train_time:.1f}s  infer={infer_ms:.3f}ms/sample")

            results.append({
                "task":        task,
                "model":       model_name,
                "accuracy":    round(acc, 4),
                "f1_weighted": round(f1, 4),
                "train_time_s": round(train_time, 2),
                "infer_ms_per_sample": round(infer_ms, 4),
                "type":        "classical",
            })

        # ── Transformer models ─────────────────────────────────────────────────
        if not skip_transformers:
            if not _transformer_available():
                print("\n   ⚠️  transformers/torch not installed.")
                print("   Run: make install-benchmark   then re-run make benchmark\n")
            else:
                transformer_models = {
                    "DistilBERT": "distilbert-base-uncased",
                    "FinBERT":    "ProsusAI/finbert",
                }
                for display_name, hf_name in transformer_models.items():
                    print(f"   Training {display_name} ({hf_name}) — this may take several minutes …")
                    try:
                        metrics = _run_transformer(
                            hf_name, X_train, X_test, y_train, y_test, task
                        )
                        print(
                            f"   acc={metrics['accuracy']:.3f}  f1={metrics['f1']:.3f}  "
                            f"train={metrics['train_time']:.1f}s  infer={metrics['infer_ms']:.3f}ms/sample"
                        )
                        results.append({
                            "task":        task,
                            "model":       display_name,
                            "accuracy":    round(metrics["accuracy"], 4),
                            "f1_weighted": round(metrics["f1"], 4),
                            "train_time_s": round(metrics["train_time"], 2),
                            "infer_ms_per_sample": round(metrics["infer_ms"], 4),
                            "type":        "transformer",
                        })
                    except Exception as exc:
                        print(f"   ERROR running {display_name}: {exc}")
        print()

    # ── Results table ─────────────────────────────────────────────────────────
    if not results:
        print("No results collected.")
        return

    df_results = pd.DataFrame(results)

    print(f"\n{'='*70}")
    print("  BENCHMARK RESULTS SUMMARY")
    print(f"{'='*70}")

    try:
        from tabulate import tabulate
        for task in df_results["task"].unique():
            sub = df_results[df_results["task"] == task].sort_values("accuracy", ascending=False)
            print(f"\n  {task.upper()} CLASSIFICATION")
            print(tabulate(
                sub[["model", "accuracy", "f1_weighted", "train_time_s", "infer_ms_per_sample"]],
                headers=["Model", "Accuracy", "F1 (weighted)", "Train (s)", "Infer (ms/sample)"],
                floatfmt=".4f",
                tablefmt="rounded_outline",
                showindex=False,
            ))
    except ImportError:
        print(df_results.to_string(index=False))

    # ── Best model per task ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  BEST MODEL PER TASK (by accuracy)")
    print(f"{'='*70}")
    for task in df_results["task"].unique():
        best = df_results[df_results["task"] == task].sort_values("accuracy", ascending=False).iloc[0]
        print(f"  {task.upper():12s} → {best['model']:40s} acc={best['accuracy']:.4f}  f1={best['f1_weighted']:.4f}")

    overall_best = (
        df_results.groupby("model")["accuracy"]
        .mean()
        .sort_values(ascending=False)
    )
    print(f"\n  OVERALL BEST (avg accuracy across all tasks):")
    for model, acc in overall_best.items():
        print(f"    {model:40s} avg_acc={acc:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_path, index=False)
    print(f"\n✅ Full results saved to {output_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark ML models for complaint classification")
    parser.add_argument("--input",             default="data/cfpb_prepared.csv",     help="Prepared CSV")
    parser.add_argument("--output",            default="data/benchmark_results.csv", help="Results CSV output")
    parser.add_argument("--sample",            type=int, default=2000,               help="Max rows (stratified). Use ≤2000 for transformer speed on CPU.")
    parser.add_argument("--skip-transformers", action="store_true",                  help="Run only classical models")
    args = parser.parse_args()
    run_benchmark(args.input, args.output, args.sample, args.skip_transformers)
