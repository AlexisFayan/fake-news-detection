"""
Google Fact Check Tools API Scraper
=====================================
Fetches fact-check claims via the Google Fact Check Claims Search API and
outputs a CSV compatible with clean_and_harmonize.py.

API docs: https://toolbox.google.com/factcheck/apis
Endpoint: https://factchecktools.googleapis.com/v1alpha1/claims:search

Prerequisites:
    export GOOGLE_FACTCHECK_API_KEY="your_key_here"
    (get a free key at https://console.developers.google.com/)

Usage:
    python google_factcheck.py --query "election fraud"
    python google_factcheck.py --queries-file queries.txt --output data/google_raw.csv
    python google_factcheck.py  # uses default broad queries

Output CSV columns (same schema as politifact_scraper.py output):
    url, statement, speaker, speaker_job, party_affiliation,
    ruling, date, context, subject, source
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

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
# API configuration
# ---------------------------------------------------------------------------

API_ENDPOINT = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
PAGE_SIZE = 100           # max allowed by the API
REQUEST_DELAY = 0.5       # seconds between API calls (API has generous rate limits)
MAX_PAGES_PER_QUERY = 10  # safety cap; API paginates via pageToken

# Default queries covering political topics similar to LIAR dataset
DEFAULT_QUERIES = [
    "president election",
    "immigration policy",
    "healthcare reform",
    "tax cuts",
    "climate change",
    "economy jobs",
    "gun control",
    "foreign policy",
    "social security",
    "crime statistics",
]

OUTPUT_COLUMNS = [
    "url", "statement", "speaker", "speaker_job", "party_affiliation",
    "ruling", "date", "context", "subject", "source",
]


# ---------------------------------------------------------------------------
# Ruling normalisation (Google returns free-text ratings from many publishers)
# ---------------------------------------------------------------------------

RULING_NORMALISATION = {
    # True-ish
    "true":            "true",
    "correct":         "true",
    "accurate":        "true",
    "verified":        "true",
    "confirmed":       "true",
    # Mostly true
    "mostly true":     "mostly-true",
    "largely true":    "mostly-true",
    "mostly accurate": "mostly-true",
    # Half-true
    "half true":       "half-true",
    "half-true":       "half-true",
    "mixed":           "half-true",
    "partly true":     "half-true",
    "partially true":  "half-true",
    # Mostly false
    "mostly false":    "mostly-false",
    "mostly incorrect":"mostly-false",
    "largely false":   "mostly-false",
    # False
    "false":           "false",
    "incorrect":       "false",
    "inaccurate":      "false",
    "wrong":           "false",
    "fabricated":      "false",
    "misleading":      "false",
    # Pants on fire
    "pants on fire":   "pants-fire",
    "ridiculous":      "pants-fire",
    "absurd":          "pants-fire",
}


def normalise_rating(raw: str) -> str:
    """Map a free-text API rating to a LIAR-compatible label (best effort)."""
    key = raw.lower().strip()
    if key in RULING_NORMALISATION:
        return RULING_NORMALISATION[key]
    # Substring matching for compound strings like "Mostly False — Check"
    for pattern, normalised in RULING_NORMALISATION.items():
        if pattern in key:
            return normalised
    return raw.lower().strip()   # return as-is if no match


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def get_api_key() -> str:
    key = os.environ.get("GOOGLE_FACTCHECK_API_KEY", "")
    if not key:
        log.error(
            "GOOGLE_FACTCHECK_API_KEY environment variable not set.\n"
            "Get a free key at https://console.developers.google.com/"
        )
        sys.exit(1)
    return key


def fetch_claims(
    session: requests.Session,
    api_key: str,
    query: str,
    language: str = "en",
    page_token: str = "",
) -> dict:
    """Make a single API call and return the JSON response."""
    params = {
        "key":          api_key,
        "query":        query,
        "languageCode": language,
        "pageSize":     PAGE_SIZE,
    }
    if page_token:
        params["pageToken"] = page_token

    resp = session.get(API_ENDPOINT, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Record parsing
# ---------------------------------------------------------------------------

def parse_claim(claim: dict) -> dict:
    """
    Parse a single claim object from the Google Fact Check API response.

    API claim structure:
    {
      "text": "Statement text",
      "claimant": "Speaker name",
      "claimDate": "2020-01-01T00:00:00Z",
      "claimReview": [{
        "publisher": {"name": "...", "site": "..."},
        "url": "...",
        "title": "...",
        "reviewDate": "...",
        "textualRating": "True",
        "languageCode": "en"
      }]
    }
    """
    record: dict = {col: "" for col in OUTPUT_COLUMNS}
    record["source"] = "google_factcheck_api"

    record["statement"] = claim.get("text", "")
    record["speaker"]   = claim.get("claimant", "")

    raw_date = claim.get("claimDate", "")
    if raw_date:
        record["date"] = raw_date[:10]   # keep YYYY-MM-DD part

    reviews = claim.get("claimReview", [])
    if reviews:
        review = reviews[0]   # use first review (often most authoritative)
        record["url"]     = review.get("url", "")
        record["context"] = review.get("title", "")

        raw_rating = review.get("textualRating", "")
        record["ruling"]  = normalise_rating(raw_rating)

        publisher = review.get("publisher", {})
        record["subject"] = publisher.get("name", "")   # publisher as subject tag

    return record


# ---------------------------------------------------------------------------
# Main fetching logic
# ---------------------------------------------------------------------------

def fetch_all_for_query(
    session: requests.Session,
    api_key: str,
    query: str,
    language: str,
) -> list[dict]:
    """Paginate through all results for a single query."""
    records: list[dict] = []
    page_token = ""
    page_count = 0

    while page_count < MAX_PAGES_PER_QUERY:
        try:
            data = fetch_claims(session, api_key, query, language, page_token)
        except requests.HTTPError as exc:
            log.warning("API error for query '%s': %s", query, exc)
            break

        claims = data.get("claims", [])
        if not claims:
            break

        for claim in claims:
            record = parse_claim(claim)
            if record["statement"]:
                records.append(record)

        page_token = data.get("nextPageToken", "")
        page_count += 1

        log.info(
            "Query '%s' — page %d: %d claims (total so far: %d)",
            query, page_count, len(claims), len(records),
        )

        if not page_token:
            break
        time.sleep(REQUEST_DELAY)

    return records


def run(
    queries: list[str],
    output_path: Path,
    language: str,
) -> None:
    api_key = get_api_key()
    session = requests.Session()
    session.headers.update({"User-Agent": "FakeNewsResearchBot/1.0 (academic NLP)"})

    all_records: list[dict] = []

    for i, query in enumerate(queries, 1):
        log.info("[%d/%d] Fetching query: '%s'", i, len(queries), query)
        records = fetch_all_for_query(session, api_key, query, language)
        all_records.extend(records)
        time.sleep(REQUEST_DELAY)

    if not all_records:
        log.warning("No records fetched. Check your API key and query terms.")
        return

    df = pd.DataFrame(all_records, columns=OUTPUT_COLUMNS + ["source"])

    # Deduplicate on statement + url
    before = len(df)
    df = df.drop_duplicates(subset=["statement", "url"])
    log.info("Deduplication: %d → %d records.", before, len(df))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info("Saved %d records to %s", len(df), output_path)
    log.info("\n--- Ruling distribution ---\n%s", df["ruling"].value_counts().to_string())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch fact-checks via Google Fact Check Tools API."
    )
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="Single search query. If not set, uses a list of default political queries.",
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=None,
        help="Text file with one query per line (overrides --query and defaults).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/google_raw.csv"),
        help="Output CSV path (default: data/google_raw.csv).",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code for the API (default: en).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.queries_file and args.queries_file.exists():
        queries = [
            line.strip() for line in args.queries_file.read_text().splitlines()
            if line.strip()
        ]
        log.info("Loaded %d queries from %s", len(queries), args.queries_file)
    elif args.query:
        queries = [args.query]
    else:
        queries = DEFAULT_QUERIES
        log.info("Using %d default political queries.", len(queries))

    run(queries=queries, output_path=args.output, language=args.language)
