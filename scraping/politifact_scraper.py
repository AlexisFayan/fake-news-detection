"""
PolitiFact Fact-Check Scraper
==============================
Scrapes fact-checks from https://www.politifact.com/factchecks/
and saves them to CSV for enriching the LIAR dataset.

Usage:
    python politifact_scraper.py --pages 50
    python politifact_scraper.py --pages 100 --output data/politifact_raw.csv
    python politifact_scraper.py --pages 50 --fetch-details  # slower, adds party/context

Output CSV columns:
    url, statement, speaker, speaker_job, party_affiliation,
    ruling, date, context, subject, source
"""

import argparse
import csv
import logging
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

    PolitiFact list page structure (2024):
      <article class="m-statement ...">
        <div class="m-statement__quote"><a href="...">statement</a></div>
        <footer class="m-statement__footer">
          <div class="m-statement__meta">
            <a href="/personalities/.../">Speaker</a>  •  job title  •  date
          </div>
          <div class="m-statement__tags"><a>tag1</a> <a>tag2</a></div>
        </footer>
        <div class="m-statement__ruling"><img alt="ruling" …/></div>
      </article>
    """
    record: dict = {col: "" for col in OUTPUT_COLUMNS}
    record["source"] = "politifact_scraped"

    # --- Statement & URL ---
    quote_div = card.find(class_=lambda c: c and "statement__quote" in c)
    if quote_div:
        link = quote_div.find("a")
        if link:
            record["statement"] = _text(link)
            href = link.get("href", "")
            record["url"] = BASE_URL + href if href.startswith("/") else href

    # --- Meta block: speaker, job, date ---
    meta_div = card.find(class_=lambda c: c and "statement__meta" in c)
    if meta_div:
        speaker_link = meta_div.find("a")
        if speaker_link:
            record["speaker"] = _text(speaker_link)

        # Remaining text after speaker link = job title and date
        meta_text = _text(meta_div)
        speaker_name = record["speaker"]
        remainder = meta_text.replace(speaker_name, "", 1).strip(" •·–-\n")

        # Date is usually the last part after a separator
        parts = [p.strip() for p in remainder.split("•") if p.strip()]
        if len(parts) >= 2:
            record["speaker_job"] = parts[0]
            record["date"] = parts[-1]
        elif len(parts) == 1:
            # Could be job or date — heuristic: if it contains digits assume date
            if any(ch.isdigit() for ch in parts[0]):
                record["date"] = parts[0]
            else:
                record["speaker_job"] = parts[0]

    # Fallback: look for explicit date element
    if not record["date"]:
        date_el = card.find(class_=lambda c: c and ("date" in c or "time" in c))
        if date_el:
            record["date"] = _text(date_el)
        time_el = card.find("time")
        if time_el:
            record["date"] = time_el.get("datetime", _text(time_el))

    # --- Ruling ---
    record["ruling"] = parse_ruling(card)

    # --- Subject / Tags ---
    tags_div = card.find(class_=lambda c: c and ("tags" in c or "subjects" in c))
    if tags_div:
        tags = [_text(a) for a in tags_div.find_all("a") if _text(a)]
        record["subject"] = ", ".join(tags)

    return record


def parse_article_details(soup: BeautifulSoup) -> dict:
    """
    Extract additional fields from an individual fact-check article page.
    Returns a dict with keys: party_affiliation, context, speaker_job (if better).
    """
    details: dict = {}

    # --- Party affiliation ---
    # Usually in a "party" span or next to the speaker name
    party_el = soup.find(class_=lambda c: c and "party" in c.lower())
    if party_el:
        details["party_affiliation"] = _text(party_el)

    # Also check meta description spans for "Democrat", "Republican", etc.
    if not details.get("party_affiliation"):
        for span in soup.find_all("span"):
            text = _text(span).lower()
            for party in ("democrat", "republican", "independent", "libertarian", "green"):
                if text == party:
                    details["party_affiliation"] = text
                    break

    # --- Context ---
    # Context appears in the article intro or a dedicated field
    context_el = soup.find(class_=lambda c: c and "context" in c.lower())
    if context_el:
        details["context"] = _text(context_el)

    if not details.get("context"):
        # Heuristic: look for "in a tweet", "in a speech", "on Facebook" etc.
        intro_el = soup.find(class_=lambda c: c and ("intro" in c or "short" in c))
        if intro_el:
            details["context"] = _text(intro_el)

    # --- Speaker job (may be more complete on article page) ---
    job_el = soup.find(class_=lambda c: c and ("job" in c or "title" in c))
    if job_el:
        details["speaker_job"] = _text(job_el)

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
        # Fallback: any article element
        cards = soup.find_all("article")

    if not cards:
        log.info("No fact-check cards found on page %d — likely end of data.", page_num)
        return []

    records = []
    for card in cards:
        record = parse_list_card(card)
        if not record["statement"]:
            continue  # skip empty cards

        if fetch_details and record["url"]:
            time.sleep(REQUEST_DELAY)
            article_soup = fetch_page(session, record["url"])
            if article_soup:
                details = parse_article_details(article_soup)
                record.update({k: v for k, v in details.items() if v})

        records.append(record)

    log.info("Page %3d — %d fact-checks scraped.", page_num, len(records))
    return records


def save_csv(records: list[dict], output_path: Path, mode: str = "w") -> None:
    """Write records to CSV. mode='a' appends, mode='w' overwrites."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = mode == "w" or not output_path.exists()
    with open(output_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS + ["source"])
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
