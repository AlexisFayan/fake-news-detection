"""
Clean and Harmonize PolitiFact Scraped Data
=============================================
Takes the raw CSV produced by politifact_scraper.py and outputs a CSV that is
plug-and-play compatible with the LIAR dataset used in fake_news_detection.ipynb.

Key transformations:
  1. Text cleaning  — same pipeline as the notebook (lowercase → URLs → digits →
                       punctuation → stopwords → lemmatize → len>2 filter)
  2. Ruling mapping — 6 PolitiFact verdicts → LIAR 6-class label → binary label
  3. Credit history — count each speaker's past rulings (barely_true_count etc.)
  4. Column alignment — produce the same 14+extras columns as the LIAR TSVs

Usage:
    python clean_and_harmonize.py --input data/politifact_raw.csv
    python clean_and_harmonize.py --input data/politifact_raw.csv \\
                                  --output data/politifact_clean.csv

Output CSV columns (LIAR-compatible + extras):
    id, label, statement, subject, speaker, speaker_job, state_info,
    party_affiliation, barely_true_count, false_count, half_true_count,
    mostly_true_count, pants_on_fire_count, context,
    source, text_clean, label_binary, ruling_raw, url, date
"""

import argparse
import logging
import re
import string
import sys
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NLTK resources (download if missing)
# ---------------------------------------------------------------------------

def ensure_nltk_resources() -> None:
    for resource, path in [
        ("stopwords",   "corpora/stopwords"),
        ("wordnet",     "corpora/wordnet"),
        ("omw-1.4",     "corpora/omw-1.4"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            log.info("Downloading NLTK resource: %s", resource)
            nltk.download(resource, quiet=True)


# ---------------------------------------------------------------------------
# Ruling → LIAR label mapping
# ---------------------------------------------------------------------------

# PolitiFact uses "Mostly False" where LIAR uses "barely-true" — both sit in
# the same difficulty tier.  All other labels map directly.
RULING_TO_LIAR_LABEL = {
    "true":         "true",
    "mostly-true":  "mostly-true",
    "half-true":    "half-true",
    "mostly-false": "barely-true",   # closest LIAR equivalent
    "barely-true":  "barely-true",   # if scraper already mapped it
    "false":        "false",
    "pants-fire":   "pants-fire",
}

# Binary: matches notebook logic (pants-fire/false/barely-true → fake)
FAKE_LABELS = {"pants-fire", "false", "barely-true"}

def ruling_to_label(ruling: str) -> str:
    """Map a normalised ruling string to a LIAR 6-class label."""
    return RULING_TO_LIAR_LABEL.get(ruling.lower().strip(), "")


def label_to_binary(label: str) -> str:
    """Map a LIAR 6-class label to 'fake' or 'real'."""
    return "fake" if label in FAKE_LABELS else "real"


# ---------------------------------------------------------------------------
# Text preprocessing  (mirrors fake_news_detection.ipynb Section 3)
# ---------------------------------------------------------------------------

def build_preprocessor():
    """Return a preprocess_text function using the same pipeline as the notebook."""
    ensure_nltk_resources()
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    punct_table = str.maketrans("", "", string.punctuation)

    def preprocess_text(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)       # remove URLs
        text = re.sub(r"\d+", "", text)                   # remove digits
        text = text.translate(punct_table)                # remove punctuation
        tokens = text.split()
        tokens = [t for t in tokens if t not in stop_words]
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        tokens = [t for t in tokens if len(t) > 2]        # keep tokens > 2 chars
        return " ".join(tokens)

    return preprocess_text


# ---------------------------------------------------------------------------
# Credit-history counts
# ---------------------------------------------------------------------------

CREDIT_COLS = [
    "barely_true_count",
    "false_count",
    "half_true_count",
    "mostly_true_count",
    "pants_on_fire_count",
]

LABEL_TO_CREDIT_COL = {
    "barely-true":  "barely_true_count",
    "false":        "false_count",
    "half-true":    "half_true_count",
    "mostly-true":  "mostly_true_count",
    "pants-fire":   "pants_on_fire_count",
    "true":         "mostly_true_count",   # lump "true" with mostly_true
}


def compute_credit_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cumulative credit history for each speaker (count of past rulings
    per category), mirroring the LIAR metadata columns.

    Note: counts are computed across the *entire scraped dataset*, not across
    time. This is an approximation; LIAR's counts cover the speaker's full
    history on PolitiFact up to dataset creation.
    """
    for col in CREDIT_COLS:
        df[col] = 0.0

    # Aggregate counts from the label column
    for label, col in LABEL_TO_CREDIT_COL.items():
        mask = df["label"] == label
        speaker_counts = df[mask].groupby("speaker").size()
        df[col] = df["speaker"].map(speaker_counts).fillna(0.0) + df[col]

    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

# LIAR-compatible column order (matches notebook's LIAR_COLUMNS)
LIAR_COLUMNS = [
    "id", "label", "statement", "subject", "speaker", "speaker_job",
    "state_info", "party_affiliation",
    "barely_true_count", "false_count", "half_true_count",
    "mostly_true_count", "pants_on_fire_count",
    "context",
]

EXTRA_COLUMNS = ["source", "text_clean", "label_binary", "ruling_raw", "url", "date"]


def clean_and_harmonize(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Full pipeline: raw CSV → LIAR-compatible clean CSV."""

    # ---- Load raw data ----
    log.info("Loading raw data from %s", input_path)
    df = pd.read_csv(input_path)
    log.info("Loaded %d rows, columns: %s", len(df), list(df.columns))

    # ---- Basic sanity check ----
    required = {"statement", "ruling", "speaker"}
    missing = required - set(df.columns)
    if missing:
        log.error("Raw CSV missing required columns: %s", missing)
        sys.exit(1)

    # ---- Drop rows with empty statement or ruling ----
    before = len(df)
    df = df[df["statement"].notna() & df["statement"].str.strip().ne("")]
    df = df[df["ruling"].notna() & df["ruling"].str.strip().ne("")]
    log.info("Dropped %d rows with empty statement/ruling.", before - len(df))

    # ---- Map ruling → LIAR label → binary ----
    df["ruling_raw"] = df["ruling"].str.strip()
    df["label"] = df["ruling_raw"].apply(ruling_to_label)

    unmapped = df[df["label"] == ""]["ruling_raw"].value_counts()
    if not unmapped.empty:
        log.warning("Unmapped rulings (will be dropped):\n%s", unmapped.to_string())
    df = df[df["label"] != ""]
    log.info("After ruling mapping: %d rows.", len(df))

    df["label_binary"] = df["label"].apply(label_to_binary)

    # ---- Text cleaning ----
    log.info("Running text preprocessing (this may take a minute)…")
    preprocess = build_preprocessor()
    df["text_clean"] = df["statement"].apply(preprocess)

    # ---- Metadata / LIAR-compatible columns ----
    df["state_info"] = ""      # not available from PolitiFact scrape

    # Fill missing optional fields
    for col in ("speaker_job", "party_affiliation", "context", "subject"):
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("").astype(str)

    df["speaker"] = df["speaker"].fillna("unknown").astype(str)

    # ---- Credit history ----
    log.info("Computing speaker credit history…")
    df = compute_credit_history(df)

    # ---- Generate IDs ----
    df["id"] = (
        "scraped_" + df.index.astype(str).str.zfill(5) + ".json"
    )

    # ---- Source tag ----
    df["source"] = "politifact_scraped"

    # ---- URL / date passthrough ----
    for col in ("url", "date"):
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("").astype(str)

    # ---- Select and reorder columns ----
    final_cols = LIAR_COLUMNS + EXTRA_COLUMNS
    df_out = df.reindex(columns=final_cols)

    # ---- Save ----
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    log.info("Saved %d clean records to %s", len(df_out), output_path)

    # ---- Summary statistics ----
    log.info("\n--- Label distribution ---\n%s", df_out["label"].value_counts().to_string())
    log.info("\n--- Binary label distribution ---\n%s", df_out["label_binary"].value_counts().to_string())

    return df_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean and harmonize scraped PolitiFact data to LIAR format."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/politifact_raw.csv"),
        help="Raw CSV from politifact_scraper.py (default: data/politifact_raw.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/politifact_clean.csv"),
        help="Output clean CSV (default: data/politifact_clean.csv).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    clean_and_harmonize(args.input, args.output)
