"""
scripts/prepare_cfpb_data.py
----------------------------
Converts a raw CFPB Consumer Complaint CSV into a clean CSV
that can be uploaded via the app's CSV upload endpoint or
used with seed_from_cfpb.py.

Usage:
    python3 scripts/prepare_cfpb_data.py \
        --input  data/cfpb_raw.csv \
        --output data/cfpb_prepared.csv \
        --limit  5000

Download the raw CSV from:
  https://www.consumerfinance.gov/data-research/consumer-complaints/
  (click "Download all complaint data")
  OR from Kaggle:
  https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# ── Category mapping ──────────────────────────────────────────────────────────
# Maps CFPB "Product" values → our three internal categories.
# Extend this dict if you want to support more categories.

CATEGORY_MAP: dict[str, str] = {
    # ── Kaggle simplified product names ───────────────────────────────────────
    "retail_banking":       "ATM_FAILURE",
    "credit_card":          "FRAUD_DETECTION",
    "debt_collection":      "FRAUD_DETECTION",
    "credit_reporting":     "UX_ISSUES",
    "mortgages_and_loans":  "UX_ISSUES",

    # ── Standard CFPB product names (consumerfinance.gov download) ────────────
    "Checking or savings account":          "ATM_FAILURE",
    "Bank account or service":              "ATM_FAILURE",
    "Prepaid card":                         "ATM_FAILURE",
    "Money transfers":                      "ATM_FAILURE",
    "Money transfer, virtual currency, or money service": "ATM_FAILURE",
    "Credit card":                          "FRAUD_DETECTION",
    "Credit card or prepaid card":          "FRAUD_DETECTION",
    "Debt collection":                      "FRAUD_DETECTION",
    "Identity theft / Fraud / Embezzlement": "FRAUD_DETECTION",
    "Payday loan":                          "FRAUD_DETECTION",
    "Payday loan, title loan, or personal loan": "FRAUD_DETECTION",
    "Mortgage":                             "UX_ISSUES",
    "Student loan":                         "UX_ISSUES",
    "Consumer loan":                        "UX_ISSUES",
    "Vehicle loan or lease":                "UX_ISSUES",
    "Credit reporting":                     "UX_ISSUES",
    "Credit reporting, credit repair services, or other personal consumer reports": "UX_ISSUES",
    "Other financial service":              "UX_ISSUES",
}

# ── Severity derivation ───────────────────────────────────────────────────────

def _derive_severity(row: pd.Series) -> str:
    """
    Infer severity from CFPB meta-columns when available,
    otherwise fall back to keyword-based heuristics on the narrative text.
    """
    disputed = str(row.get("Consumer disputed?", "")).strip().lower()
    timely   = str(row.get("Timely response?",   "")).strip().lower()

    # Meta-column path (full CFPB download)
    if disputed == "yes" and timely == "no":
        return "critical"
    if disputed == "yes":
        return "high"
    if timely == "no":
        return "medium"

    # Keyword fallback (Kaggle simplified dataset has no meta-columns)
    text = str(row.get("narrative", row.get("Consumer complaint narrative", ""))).lower()
    if any(w in text for w in (
        "fraud", "stolen", "unauthorized", "identity theft", "scam",
        "criminal", "illegal", "stole", "hack",
    )):
        return "critical"
    if any(w in text for w in (
        "urgent", "immediately", "emergency", "cannot access", "locked out",
        "suspended", "blocked", "threatening", "lawsuit",
    )):
        return "high"
    if any(w in text for w in (
        "weeks", "months", "repeated", "multiple times", "still waiting",
        "no response", "ignored", "unresolved", "never",
    )):
        return "medium"
    return "low"


# ── Emotion heuristics ────────────────────────────────────────────────────────

def _derive_emotion(text: str) -> str:
    """
    Very lightweight keyword scan to assign a dominant emotion.
    The classifier will re-predict this after training — this is only
    used during the initial seed so the field is never blank.
    """
    text_lower = text.lower()
    if any(w in text_lower for w in ("fraud", "steal", "unauthorized", "scam", "stolen")):
        return "angry"
    if any(w in text_lower for w in ("disappointed", "terrible", "awful", "horrible")):
        return "frustrated"
    if any(w in text_lower for w in ("confus", "don't understand", "unclear")):
        return "confused"
    if any(w in text_lower for w in ("please", "help", "need", "urgent")):
        return "anxious"
    return "frustrated"


# ── Main ──────────────────────────────────────────────────────────────────────

def prepare(input_path: str, output_path: str, limit: int) -> None:
    print(f"Reading {input_path} …")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"  Raw rows: {len(df):,}")

    # ── 1. Keep only rows with a real narrative ───────────────────────────────
    narrative_col = next(
        (c for c in df.columns if c.lower() in ("narrative", "consumer_complaint_narrative")
         or "narrative" in c.lower() or "complaint" in c.lower()),
        None,
    )
    if narrative_col is None:
        sys.exit("ERROR: Could not find a narrative/complaint text column. "
                 "Check the CSV columns with: python3 -c \"import pandas as pd; "
                 "print(pd.read_csv('data/cfpb_raw.csv', nrows=0).columns.tolist())\"")

    df = df.dropna(subset=[narrative_col])
    df = df[df[narrative_col].str.strip().str.len() >= 50]
    print(f"  After dropping nulls/short narratives: {len(df):,}")

    # ── 2. Map category ───────────────────────────────────────────────────────
    product_col = next((c for c in df.columns if "product" in c.lower()), None)
    if product_col:
        df["category"] = df[product_col].map(CATEGORY_MAP)
        df = df.dropna(subset=["category"])   # drop unmapped products
    else:
        df["category"] = "UX_ISSUES"          # fallback if no product column
    print(f"  After category mapping: {len(df):,}")

    # ── 3. Derive severity ────────────────────────────────────────────────────
    df["severity"] = df.apply(_derive_severity, axis=1)

    # ── 4. Derive emotion ─────────────────────────────────────────────────────
    df["emotion"] = df[narrative_col].apply(_derive_emotion)

    # ── 5. Map remaining columns ──────────────────────────────────────────────
    out = pd.DataFrame()
    out["complaint_text"] = df[narrative_col].str.strip()
    out["category"]       = df["category"]
    out["severity"]       = df["severity"]
    out["emotion"]        = df["emotion"]

    # Optional enrichment columns (present in CFPB, silently skipped if absent)
    col_map = {
        "customer_name": ["Consumer name", "consumer_name"],
        "location":      ["State", "state"],
        "channel":       ["Submitted via", "submitted_via"],
        "customer_id":   ["Complaint ID", "complaint_id"],
        "specific_issue": ["Issue", "issue"],
    }
    for out_col, candidates in col_map.items():
        for cand in candidates:
            if cand in df.columns:
                out[out_col] = df[cand].fillna("").astype(str)
                break

    # ── 6. Sample / limit ─────────────────────────────────────────────────────
    if limit and len(out) > limit:
        # Stratified sample by category so classes stay balanced
        per_class = limit // out["category"].nunique()
        out = (
            out.groupby("category", group_keys=False)
               .apply(lambda g: g.sample(min(len(g), per_class), random_state=42))
               .reset_index(drop=True)
        )
        print(f"  Sampled to {len(out):,} rows (stratified by category)")

    # ── 7. Final clean ────────────────────────────────────────────────────────
    out = out.drop_duplicates(subset=["complaint_text"])
    out = out[out["complaint_text"].str.len().between(50, 2000)]
    # Truncate very long texts to 2000 chars to fit API limits
    out["complaint_text"] = out["complaint_text"].str[:2000]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f"\n✅ Prepared {len(out):,} complaints → {output_path}")
    print(f"   Category breakdown:\n{out['category'].value_counts().to_string()}")
    print(f"   Severity breakdown:\n{out['severity'].value_counts().to_string()}")
    print(f"\nNext step: python3 scripts/seed_from_cfpb.py --input {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare CFPB data for the complaint system")
    parser.add_argument("--input",  default="data/cfpb_raw.csv",      help="Path to raw CFPB CSV")
    parser.add_argument("--output", default="data/cfpb_prepared.csv", help="Output path")
    parser.add_argument("--limit",  type=int, default=5000,           help="Max rows (0 = no limit)")
    args = parser.parse_args()
    prepare(args.input, args.output, args.limit)
