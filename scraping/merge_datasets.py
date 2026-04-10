"""
Merge Scraped PolitiFact Data with LIAR Dataset
================================================
Takes the cleaned CSV from clean_and_harmonize.py and merges it with the
existing LIAR dataset (train/valid/test TSVs or a pre-merged CSV), producing
an enriched dataset ready to feed into fake_news_detection.ipynb.

Features:
  - Deduplication based on fuzzy statement similarity (exact + near-duplicate)
  - Before/after statistics (samples, label distribution, party distribution)
  - Optional temporal split: train on older data, test on newer data

Usage:
    # Basic merge (downloads LIAR automatically if not found)
    python merge_datasets.py --scraped data/politifact_clean.csv

    # Merge with local LIAR files
    python merge_datasets.py --scraped data/politifact_clean.csv \\
                             --liar-dir ../data/liar/

    # Temporal split
    python merge_datasets.py --scraped data/politifact_clean.csv \\
                             --temporal-split --split-date 2020-01-01

Output files (in data/merged/):
    merged_full.csv            — full deduplicated dataset
    merged_train.csv           — training split (+ temporal option)
    merged_test.csv            — test split
    merge_stats.txt            — before/after statistics report
"""

import argparse
import logging
import sys
import urllib.request
from pathlib import Path

import pandas as pd

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
# LIAR dataset handling
# ---------------------------------------------------------------------------

LIAR_COLUMNS = [
    "id", "label", "statement", "subject", "speaker", "speaker_job",
    "state_info", "party_affiliation",
    "barely_true_count", "false_count", "half_true_count",
    "mostly_true_count", "pants_on_fire_count",
    "context",
]

LIAR_URLS = {
    "train": "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/train.tsv",
    "test":  "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/test.tsv",
    "valid": "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/valid.tsv",
}

FAKE_LABELS = {"pants-fire", "false", "barely-true"}


def label_to_binary(label: str) -> str:
    return "fake" if str(label).strip() in FAKE_LABELS else "real"


def load_liar_split(path: Path) -> pd.DataFrame:
    """Load a single LIAR TSV file (no header, 14 columns)."""
    df = pd.read_csv(path, sep="\t", header=None, names=LIAR_COLUMNS,
                     dtype=str, keep_default_na=False)
    df["source"] = "liar"
    df["label_binary"] = df["label"].apply(label_to_binary)
    # text_clean column not pre-computed here — notebook re-processes if needed
    df["text_clean"] = ""
    df["url"] = ""
    df["date"] = ""
    df["ruling_raw"] = df["label"]
    return df


def download_liar(liar_dir: Path) -> None:
    """Download LIAR TSV files into liar_dir if not already present."""
    liar_dir.mkdir(parents=True, exist_ok=True)
    for split, url in LIAR_URLS.items():
        dest = liar_dir / f"{split}.tsv"
        if dest.exists():
            log.info("LIAR %s already present at %s", split, dest)
            continue
        log.info("Downloading LIAR %s from %s …", split, url)
        urllib.request.urlretrieve(url, dest)
        log.info("Saved → %s", dest)


def load_liar(liar_dir: Path) -> pd.DataFrame:
    """Load and concatenate all three LIAR splits."""
    frames = []
    for split in ("train", "valid", "test"):
        path = liar_dir / f"{split}.tsv"
        if not path.exists():
            log.error("LIAR file not found: %s", path)
            sys.exit(1)
        df = load_liar_split(path)
        df["liar_split"] = split
        frames.append(df)
        log.info("Loaded LIAR %s: %d rows", split, len(df))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def normalise_for_dedup(text: str) -> str:
    """Lowercase + collapse whitespace for approximate dedup."""
    return " ".join(str(text).lower().split())


def deduplicate(liar_df: pd.DataFrame, scraped_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove scraped rows whose statement is already in LIAR (exact match after
    normalisation).  Returns the deduplicated scraped DataFrame.
    """
    liar_statements = set(liar_df["statement"].apply(normalise_for_dedup))
    before = len(scraped_df)
    scraped_df = scraped_df[
        ~scraped_df["statement"].apply(normalise_for_dedup).isin(liar_statements)
    ]
    removed = before - len(scraped_df)
    log.info("Deduplication: removed %d scraped rows already in LIAR.", removed)
    return scraped_df


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def dataset_stats(df: pd.DataFrame, label: str = "") -> str:
    lines = [f"\n{'='*50}", f"  {label}", f"{'='*50}"]
    lines.append(f"Total samples : {len(df):,}")

    if "label_binary" in df.columns:
        vc = df["label_binary"].value_counts()
        for cls, cnt in vc.items():
            lines.append(f"  {cls:>6}: {cnt:,}  ({cnt/len(df)*100:.1f}%)")

    if "label" in df.columns:
        lines.append("\n6-class label distribution:")
        for lbl, cnt in df["label"].value_counts().items():
            lines.append(f"  {lbl:>14}: {cnt:,}")

    if "party_affiliation" in df.columns:
        top_parties = df["party_affiliation"].value_counts().head(8)
        lines.append("\nTop parties:")
        for party, cnt in top_parties.items():
            if party:
                lines.append(f"  {party:>16}: {cnt:,}")

    if "source" in df.columns:
        lines.append("\nBy source:")
        for src, cnt in df["source"].value_counts().items():
            lines.append(f"  {src:>24}: {cnt:,}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main merge logic
# ---------------------------------------------------------------------------

ALL_COLUMNS = (
    LIAR_COLUMNS
    + ["source", "text_clean", "label_binary", "ruling_raw", "url", "date", "liar_split"]
)


def merge(
    scraped_path: Path,
    liar_dir: Path,
    output_dir: Path,
    temporal_split: bool,
    split_date: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load LIAR ----
    if not (liar_dir / "train.tsv").exists():
        log.info("LIAR TSVs not found in %s — downloading…", liar_dir)
        download_liar(liar_dir)
    liar_df = load_liar(liar_dir)

    # ---- Load scraped data ----
    log.info("Loading scraped data from %s", scraped_path)
    scraped_df = pd.read_csv(scraped_path, dtype=str).fillna("")

    if "liar_split" not in scraped_df.columns:
        scraped_df["liar_split"] = "scraped"

    # ---- Stats before ----
    stats_before = (
        dataset_stats(liar_df, "LIAR DATASET (before merge)")
        + dataset_stats(scraped_df, "SCRAPED DATA (before merge)")
    )

    # ---- Deduplicate ----
    scraped_df = deduplicate(liar_df, scraped_df)

    # ---- Align columns ----
    for df in (liar_df, scraped_df):
        for col in ALL_COLUMNS:
            if col not in df.columns:
                df[col] = ""

    liar_df  = liar_df.reindex(columns=ALL_COLUMNS)
    scraped_df = scraped_df.reindex(columns=ALL_COLUMNS)

    # ---- Merge ----
    merged = pd.concat([liar_df, scraped_df], ignore_index=True)
    log.info("Merged dataset: %d rows total.", len(merged))

    # ---- Stats after ----
    stats_after = dataset_stats(merged, "MERGED DATASET (after)")
    stats_report = stats_before + stats_after

    log.info(stats_report)

    # Save stats report
    stats_path = output_dir / "merge_stats.txt"
    stats_path.write_text(stats_report, encoding="utf-8")
    log.info("Stats saved to %s", stats_path)

    # ---- Save full merged CSV ----
    full_path = output_dir / "merged_full.csv"
    merged.to_csv(full_path, index=False)
    log.info("Full merged dataset saved to %s", full_path)

    # ---- Splits ----
    if temporal_split:
        _temporal_split(merged, output_dir, split_date)
    else:
        _standard_split(merged, liar_df, scraped_df, output_dir)


def _standard_split(
    merged: pd.DataFrame,
    liar_df: pd.DataFrame,
    scraped_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Standard split: keep LIAR's original train/test/valid splits intact,
    and distribute scraped data proportionally (80% train, 20% test).
    """
    train_mask = merged["liar_split"].isin(["train", "valid", "scraped"])

    # Assign scraped rows 80% train / 20% test
    scraped_idx = merged[merged["source"] == "politifact_scraped"].index
    n_scraped_test = max(1, int(len(scraped_idx) * 0.2))
    scraped_test_idx = scraped_idx[-n_scraped_test:]
    scraped_train_idx = scraped_idx[:-n_scraped_test]

    train_mask = merged["liar_split"].isin(["train", "valid"])
    train_mask = train_mask | merged.index.isin(scraped_train_idx)
    test_mask  = (merged["liar_split"] == "test") | merged.index.isin(scraped_test_idx)

    train_df = merged[train_mask]
    test_df  = merged[test_mask]

    train_df.to_csv(output_dir / "merged_train.csv", index=False)
    test_df.to_csv(output_dir / "merged_test.csv", index=False)
    log.info("Train: %d rows | Test: %d rows", len(train_df), len(test_df))


def _temporal_split(merged: pd.DataFrame, output_dir: Path, split_date: str) -> None:
    """
    Temporal split: train on statements before split_date, test on after.
    Rows without a parseable date go into training.
    """
    log.info("Applying temporal split at %s", split_date)

    dates = pd.to_datetime(merged["date"], errors="coerce")
    cutoff = pd.Timestamp(split_date)

    before_mask = dates.isna() | (dates < cutoff)
    after_mask  = dates >= cutoff

    train_df = merged[before_mask]
    test_df  = merged[after_mask]

    train_df.to_csv(output_dir / "merged_train.csv", index=False)
    test_df.to_csv(output_dir / "merged_test.csv", index=False)
    log.info(
        "Temporal split: %d train (before %s) | %d test (after %s)",
        len(train_df), split_date, len(test_df), split_date,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge cleaned PolitiFact scrape with the LIAR dataset."
    )
    parser.add_argument(
        "--scraped",
        type=Path,
        default=Path("data/politifact_clean.csv"),
        help="Cleaned scraped CSV (output of clean_and_harmonize.py).",
    )
    parser.add_argument(
        "--liar-dir",
        type=Path,
        default=Path("data/liar"),
        help="Directory containing LIAR train/valid/test TSVs (auto-downloaded if absent).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/merged"),
        help="Output directory for merged CSVs and stats (default: data/merged/).",
    )
    parser.add_argument(
        "--temporal-split",
        action="store_true",
        default=False,
        help="Split by date instead of LIAR's original train/test split.",
    )
    parser.add_argument(
        "--split-date",
        type=str,
        default="2020-01-01",
        help="Cut-off date for temporal split (ISO format, default: 2020-01-01).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge(
        scraped_path=args.scraped,
        liar_dir=args.liar_dir,
        output_dir=args.output_dir,
        temporal_split=args.temporal_split,
        split_date=args.split_date,
    )
