"""
PolitiFact Fact-Check Scraper
==============================
Scrapes fact-checks from https://www.politifact.com/factchecks/
and saves them to CSV for enriching the LIAR dataset.

Usage:
    python politifact_scraper.py --pages 50
    python politifact_scraper.py --pages 100 --output data/politifact_raw.csv
    python politifact_scraper.py --pages 50 --fetch-details  # slower, adds subjects + speaker_job

Output CSV columns:
    url, statement, speaker, speaker_job, party_affiliation,
    ruling, date, context, subject, source

HTML structure notes (verified April 2026):
  List page card  → article.m-statement
    a.m-statement__name         → speaker name
    div.m-statement__desc       → "stated on {Month D, YYYY} {context}:"
    div.m-statement__quote a    → statement text + URL
    div.m-statement__meter img  → ruling (alt attribute)
    footer.m-statement__footer  → "By {PolitiFact author} • {pub date}" (NOT speaker job)
  Article page  → ul.m-list--horizontal a.c-tag  → subject tags (excl. personality links)
  Personality page → div.m-pageheader__body p    → bio text (contains speaker job)
"""

import argparse
import csv
import logging
import re
import sys
import time
import urllib.robotparser
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://www.politifact.com"
FACTCHECKS_URL = f"{BASE_URL}/factchecks/"
USER_AGENT = "FakeNewsResearchBot/1.0 (academic NLP project; contact: research@example.com)"
REQUEST_DELAY = 1.5       # seconds between requests
REQUEST_TIMEOUT = 20      # seconds
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0       # exponential backoff multiplier
SAVE_INTERVAL = 50        # save CSV every N pages

# Ruling normalisation: alt-text / image filename → LIAR-compatible label
RULING_MAP = {
    "true":          "true",
    "mostly-true":   "mostly-true",
    "mostly true":   "mostly-true",
    "half-true":     "half-true",
    "half true":     "half-true",
    "barely-true":   "barely-true",
    "barely true":   "barely-true",
    "mostly-false":  "mostly-false",
    "mostly false":  "mostly-false",
    "false":         "false",
    "pants-fire":    "pants-fire",
    "pants on fire": "pants-fire",
    "pants-on-fire": "pants-fire",
}

OUTPUT_COLUMNS = [
    "url", "statement", "speaker", "speaker_job", "party_affiliation",
    "ruling", "date", "context", "subject", "source",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("scraper.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Robots.txt check
# ---------------------------------------------------------------------------

def check_robots_txt() -> bool:
    """Return True if /factchecks/ is allowed by robots.txt."""
    rp = urllib.robotparser.RobotFileParser()
    robots_url = f"{BASE_URL}/robots.txt"
    try:
        rp.set_url(robots_url)
        rp.read()
        allowed = rp.can_fetch(USER_AGENT, FACTCHECKS_URL)
        if allowed:
            log.info("robots.txt check passed — /factchecks/ is allowed.")
        else:
            log.warning("robots.txt disallows scraping /factchecks/. Aborting.")
        return allowed
    except Exception as exc:
        log.warning("Could not fetch robots.txt (%s). Proceeding anyway.", exc)
        return True


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def fetch_page(session: requests.Session, url: str) -> Optional[BeautifulSoup]:
    """Fetch a URL with retries and exponential backoff. Returns BeautifulSoup or None."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return BeautifulSoup(resp.text, "html.parser")
            if resp.status_code == 404:
                log.info("404 on %s — end of pagination.", url)
                return None
            log.warning("HTTP %s on %s (attempt %d/%d)", resp.status_code, url, attempt, MAX_RETRIES)
        except requests.RequestException as exc:
            log.warning("Request error on %s (attempt %d/%d): %s", url, attempt, MAX_RETRIES, exc)
        if attempt < MAX_RETRIES:
            sleep_time = REQUEST_DELAY * (RETRY_BACKOFF ** (attempt - 1))
            time.sleep(sleep_time)
    log.error("Failed to fetch %s after %d attempts.", url, MAX_RETRIES)
    return None


# ---------------------------------------------------------------------------
# HTML parsing helpers
# ---------------------------------------------------------------------------

def _text(element) -> str:
    """Return stripped text from a BS4 element, or empty string."""
    return element.get_text(separator=" ", strip=True) if element else ""


def _normalise_ruling(raw: str) -> str:
    """Normalise a raw ruling string to a LIAR-compatible label."""
    key = raw.lower().strip().replace("_", "-")
    return RULING_MAP.get(key, raw.lower().strip())


def parse_ruling(card) -> str:
    """Extract ruling from a fact-check card using multiple selector strategies."""
    # Strategy 1: ruling image alt text
    for img in card.find_all("img"):
        alt = img.get("alt", "").strip()
        if alt:
            normalised = _normalise_ruling(alt)
            if normalised in RULING_MAP.values():
                return normalised
        # Strategy 2: ruling embedded in image src filename
        src = img.get("src", "")
        for key in RULING_MAP:
            if key in src.lower():
                return RULING_MAP[key]

    # Strategy 3: dedicated ruling element class
    ruling_el = card.find(class_=lambda c: c and "ruling" in c.lower())
    if ruling_el:
        return _normalise_ruling(_text(ruling_el))

    return ""


def parse_list_card(card) -> dict:
    """
    Parse a single fact-check card from the /factchecks/ list page.

    Verified HTML structure (April 2026):
      article.m-statement
        div.m-statement__meta
          a.m-statement__name          → speaker name
          div.m-statement__desc        → "stated on {Month D, YYYY} {context}:"
        div.m-statement__quote
          a[href]                      → statement text + URL
        div.m-statement__meter
          img[alt]                     → ruling label
        footer.m-statement__footer     → "By {author} • {pub_date}"  (NOT speaker job)
    """
    record: dict = {col: "" for col in OUTPUT_COLUMNS}
    record["source"] = "politifact_scraped"

    # --- Statement & URL ---
    quote_div = card.find(class_="m-statement__quote")
    if quote_div:
        link = quote_div.find("a")
        if link:
            record["statement"] = _text(link)
            href = link.get("href", "")
            record["url"] = BASE_URL + href if href.startswith("/") else href

    # --- Speaker ---
    speaker_link = card.find("a", class_="m-statement__name")
    if speaker_link:
        record["speaker"] = speaker_link.get_text(strip=True)

    # --- Date + Context from m-statement__desc ---
    # Format: "stated on {Month D, YYYY} {context}:"
    desc_el = card.find(class_="m-statement__desc")
    if desc_el:
        desc = desc_el.get_text(separator=" ", strip=True)
        # Extract date (Month D, YYYY) and context (everything after)
        m = re.match(
            r'stated on\s+(\w+\s+\d+,\s+\d{4})\s+(.*?):\s*$',
            desc, re.IGNORECASE
        )
        if m:
            record["date"]    = m.group(1).strip()   # e.g. "April 6, 2026"
            record["context"] = m.group(2).strip()   # e.g. "in social media posts"
        else:
            # Fallback: pull any "Month D, YYYY" pattern
            date_m = re.search(r'(\w+\s+\d+,\s+\d{4})', desc)
            if date_m:
                record["date"] = date_m.group(1)
            record["context"] = desc  # store full desc as context if regex fails

    # --- Ruling ---
    record["ruling"] = parse_ruling(card)

    # subject and speaker_job are only available via --fetch-details
    return record


def parse_article_details(session: requests.Session, article_url: str, speaker_link: str) -> dict:
    """
    Fetch an individual article page and its speaker personality page to extract:
      - subject  : topic tags (from ul.m-list--horizontal a.c-tag, excl. personality links)
      - speaker_job : first sentence of speaker bio on personality page

    party_affiliation is not available in structured form on PolitiFact — left empty.
    """
    details: dict = {}

    # --- Article page: subject tags ---
    soup = fetch_page(session, article_url)
    if soup:
        tags_list = soup.find("ul", class_="m-list--horizontal")
        if tags_list:
            subjects = []
            for a in tags_list.find_all("a", class_="c-tag"):
                href = a.get("href", "")
                if "/personalities/" in href:
                    continue   # skip speaker self-tag
                label = a.get("title") or a.get_text(strip=True)
                if label:
                    subjects.append(label)
            if subjects:
                details["subject"] = ", ".join(subjects)

    # --- Personality page: speaker job ---
    if speaker_link:
        personality_url = (
            BASE_URL + speaker_link if speaker_link.startswith("/") else speaker_link
        )
        time.sleep(REQUEST_DELAY)
        psoup = fetch_page(session, personality_url)
        if psoup:
            bio_el = psoup.find(class_="m-pageheader__body")
            if bio_el:
                bio_p = bio_el.find("p")
                if bio_p:
                    bio_text = bio_p.get_text(strip=True)
                    # Bio is "Name is the {job title}." — extract everything after "is "
                    job_m = re.search(r'\bis\s+(?:the\s+|an?\s+)?(.+?)(?:\s+and\b|\.|,|$)', bio_text, re.IGNORECASE)
                    if job_m:
                        details["speaker_job"] = job_m.group(1).strip()

    return details


# ---------------------------------------------------------------------------
# Main scraping logic
# ---------------------------------------------------------------------------

def scrape_page(session: requests.Session, page_num: int, fetch_details: bool) -> list[dict]:
    """Scrape one list page and return a list of record dicts."""
    url = f"{FACTCHECKS_URL}?page={page_num}"
    soup = fetch_page(session, url)
    if soup is None:
        return []

    cards = soup.find_all("article", class_=lambda c: c and "m-statement" in c)
    if not cards:
        log.info("No fact-check cards found on page %d — likely end of data.", page_num)
        return []

    records = []
    for card in cards:
        record = parse_list_card(card)
        if not record["statement"]:
            continue  # skip empty/malformed cards

        if fetch_details and record["url"]:
            # Grab speaker personality href for job extraction
            speaker_a = card.find("a", class_="m-statement__name")
            speaker_href = speaker_a.get("href", "") if speaker_a else ""
            time.sleep(REQUEST_DELAY)
            details = parse_article_details(session, record["url"], speaker_href)
            record.update({k: v for k, v in details.items() if v})

        records.append(record)

    log.info("Page %3d — %d fact-checks scraped.", page_num, len(records))
    return records


def save_csv(records: list[dict], output_path: Path, mode: str = "w") -> None:
    """Write records to CSV. mode='a' appends, mode='w' overwrites."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = mode == "w" or not output_path.exists()
    with open(output_path, mode, newline="", encoding="utf-8") as f:
        # OUTPUT_COLUMNS already includes "source" — do not append it again
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(records)


def run_scraper(num_pages: int, output_path: Path, fetch_details: bool) -> None:
    """Main entry point: scrape num_pages pages and save to output_path."""
    if not check_robots_txt():
        sys.exit(1)

    session = make_session()
    all_records: list[dict] = []
    first_save = True

    for page in range(1, num_pages + 1):
        records = scrape_page(session, page, fetch_details)

        if not records:
            log.info("Empty page %d — stopping early.", page)
            break

        all_records.extend(records)

        # Incremental save every SAVE_INTERVAL pages
        if page % SAVE_INTERVAL == 0:
            mode = "w" if first_save else "a"
            save_csv(all_records if first_save else records, output_path, mode=mode)
            log.info("Incremental save: %d total records → %s", len(all_records), output_path)
            if first_save:
                first_save = False
                all_records = []  # free memory, already saved

        time.sleep(REQUEST_DELAY)

    # Final save (remaining records not yet saved)
    if all_records:
        mode = "w" if first_save else "a"
        save_csv(all_records, output_path, mode=mode)

    log.info("Done. Total records saved to %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape PolitiFact fact-checks for the Fake News Detection project."
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=50,
        help="Number of list pages to scrape (default: 50, ~10–15 fact-checks/page).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/politifact_raw.csv"),
        help="Output CSV path (default: data/politifact_raw.csv).",
    )
    parser.add_argument(
        "--fetch-details",
        action="store_true",
        default=False,
        help="Also fetch individual article pages for party_affiliation and context (slower).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    log.info("Starting PolitiFact scraper: %d pages → %s", args.pages, args.output)
    run_scraper(
        num_pages=args.pages,
        output_path=args.output,
        fetch_details=args.fetch_details,
    )
