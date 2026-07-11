"""
scripts/seed_from_cfpb.py
--------------------------
Reads the prepared CFPB CSV (output of prepare_cfpb_data.py) and
inserts rows directly into MongoDB Atlas — no API call needed.

Usage:
    python3 scripts/seed_from_cfpb.py \
        --input data/cfpb_prepared.csv \
        --batch 500

Run prepare_cfpb_data.py first to create the prepared CSV.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import os
os.environ.setdefault("MODEL_DIR",           "models")
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///./mlruns.db")
os.environ.setdefault("SEEDS_DIR",           "data/seeds")

import pandas as pd

from core.database import ensure_indexes, get_db
from core.db_models import new_complaint


def seed(input_path: str, batch_size: int) -> None:
    if not Path(input_path).exists():
        sys.exit(
            f"ERROR: {input_path} not found.\n"
            "Run first: python3 scripts/prepare_cfpb_data.py"
        )

    print(f"Reading {input_path} …")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"  Rows to insert: {len(df):,}")

    # Ensure indexes exist
    ensure_indexes()

    db = get_db()
    existing = db.complaints.count_documents({})
    print(f"  Existing complaints in DB: {existing:,}")

    inserted = 0
    skipped  = 0

    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start : start + batch_size]
        docs  = []

        for _, row in chunk.iterrows():
            text = str(row.get("complaint_text", "")).strip()
            if len(text) < 20:
                skipped += 1
                continue

            doc = new_complaint(
                complaint_text    = text[:2000],
                category          = str(row.get("category",      "UX_ISSUES")),
                severity          = str(row.get("severity",       "medium")),
                emotion           = str(row.get("emotion",        "frustrated")),
                specific_issue    = str(row.get("specific_issue", "")) or None,
                location          = str(row.get("location",       "")) or None,
                channel           = str(row.get("channel",        "")) or None,
                customer_id       = str(row.get("customer_id",    "")) or None,
                customer_name     = str(row.get("customer_name",  "")) or None,
                source            = "csv_upload",
                status            = "open",
                generation_method = "cfpb_dataset",
                word_count        = len(text.split()),
                character_count   = len(text),
            )
            docs.append(doc)

        if docs:
            db.complaints.insert_many(docs)
        inserted += len(docs)

        pct = (start + len(chunk)) / len(df) * 100
        print(f"  [{pct:5.1f}%] Inserted {inserted:,} rows so far …", end="\r")

    print(f"\n\n✅ Done — inserted {inserted:,} complaints, skipped {skipped} invalid rows.")
    print(f"   Total complaints in DB: {existing + inserted:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed MongoDB from prepared CFPB CSV")
    parser.add_argument("--input", default="data/cfpb_prepared.csv", help="Prepared CSV path")
    parser.add_argument("--batch", type=int, default=500,            help="Insert batch size")
    args = parser.parse_args()
    seed(args.input, args.batch)
