"""
scripts/seed_from_cfpb.py
--------------------------
Reads the prepared CFPB CSV (output of prepare_cfpb_data.py) and
inserts rows directly into the SQLite database — no API call needed.

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

# ── Must set env vars BEFORE importing anything from core ────────────────────
import os
os.environ.setdefault("DATABASE_URL",        "sqlite:///./data/complaints.db")
os.environ.setdefault("MODEL_DIR",           "models")
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///./mlruns.db")
os.environ.setdefault("SEEDS_DIR",           "data/seeds")

import pandas as pd
from sqlalchemy.orm import Session

from core.database import SessionLocal, create_tables
from core.db_models import Complaint


def seed(input_path: str, batch_size: int) -> None:
    if not Path(input_path).exists():
        sys.exit(
            f"ERROR: {input_path} not found.\n"
            "Run first: python3 scripts/prepare_cfpb_data.py"
        )

    print(f"Reading {input_path} …")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"  Rows to insert: {len(df):,}")

    # Ensure tables exist
    create_tables()

    db: Session = SessionLocal()
    existing = db.query(Complaint).count()
    print(f"  Existing complaints in DB: {existing:,}")

    inserted = 0
    skipped  = 0

    try:
        for start in range(0, len(df), batch_size):
            chunk = df.iloc[start : start + batch_size]
            rows  = []

            for _, row in chunk.iterrows():
                text = str(row.get("complaint_text", "")).strip()
                if len(text) < 20:
                    skipped += 1
                    continue

                complaint = Complaint(
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
                rows.append(complaint)

            db.bulk_save_objects(rows)
            db.commit()
            inserted += len(rows)

            pct = (start + len(chunk)) / len(df) * 100
            print(f"  [{pct:5.1f}%] Inserted {inserted:,} rows so far …", end="\r")

    except Exception as exc:
        db.rollback()
        print(f"\nERROR during insert: {exc}")
        raise
    finally:
        db.close()

    print(f"\n\n✅ Done — inserted {inserted:,} complaints, skipped {skipped} invalid rows.")
    print(f"   Total complaints in DB: {existing + inserted:,}")
    print("\nNext step: make api → Admin Train page → Train Model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed DB from prepared CFPB CSV")
    parser.add_argument("--input", default="data/cfpb_prepared.csv", help="Prepared CSV path")
    parser.add_argument("--batch", type=int, default=500,            help="Insert batch size")
    args = parser.parse_args()
    seed(args.input, args.batch)
