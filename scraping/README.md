# Scraping Pipeline — Fake News Detection

Scripts to enrich the LIAR dataset with fresh fact-checks from PolitiFact and the Google Fact Check API.

## Structure

```
scraping/
├── politifact_scraper.py   # Scraper for politifact.com/factchecks/
├── google_factcheck.py     # Google Fact Check Tools API client
├── clean_and_harmonize.py  # Text cleaning + LIAR-compatible output
├── merge_datasets.py       # Merge scraped data with LIAR
├── requirements.txt
├── .gitignore              # Excludes data/ folder from git
└── data/                   # Created at runtime (gitignored)
    ├── politifact_raw.csv
    ├── politifact_clean.csv
    ├── google_raw.csv
    ├── liar/               # Auto-downloaded LIAR TSVs
    └── merged/             # Final merged datasets
```

---

## Installation

```bash
pip install -r scraping/requirements.txt
```

All scripts must be run from the `scraping/` directory:

```bash
cd scraping/
```

---

## 1. Scrape PolitiFact

```bash
# Scrape 50 pages (~500–750 fact-checks)
python politifact_scraper.py --pages 50

# Scrape more and fetch party/context from individual article pages (slower)
python politifact_scraper.py --pages 200 --fetch-details

# Custom output path
python politifact_scraper.py --pages 100 --output data/my_raw.csv
```

**Output:** `data/politifact_raw.csv` with columns:
`url, statement, speaker, speaker_job, party_affiliation, ruling, date, context, subject, source`

The scraper:
- Checks `robots.txt` before starting
- Sleeps 1.5 s between requests
- Retries failed requests up to 3 times (exponential backoff)
- Saves incrementally every 50 pages so no data is lost on crash
- Writes a `scraper.log` file

---

## 2. Clean & Harmonize

```bash
python clean_and_harmonize.py --input data/politifact_raw.csv
```

**Output:** `data/politifact_clean.csv` — LIAR-compatible columns:

| Column | Description |
|--------|-------------|
| `id` | Generated ID (`scraped_00001.json`) |
| `label` | 6-class LIAR label (`true`, `mostly-true`, `half-true`, `barely-true`, `false`, `pants-fire`) |
| `statement` | Original statement text |
| `subject` | Topic tags |
| `speaker` | Speaker name |
| `speaker_job` | Job title |
| `state_info` | Empty (not available from scrape) |
| `party_affiliation` | Political party |
| `barely_true_count` … `pants_on_fire_count` | Speaker's historical ruling counts (computed from scraped data) |
| `context` | Statement context |
| `source` | `"politifact_scraped"` |
| `text_clean` | Preprocessed text (same pipeline as notebook) |
| `label_binary` | `"fake"` or `"real"` |

**Ruling → binary mapping:**

| PolitiFact ruling | LIAR label | Binary |
|---|---|---|
| True | true | real |
| Mostly True | mostly-true | real |
| Half True | half-true | real |
| Mostly False | barely-true | fake |
| False | false | fake |
| Pants on Fire | pants-fire | fake |

---

## 3. Merge with LIAR

```bash
# Standard merge (LIAR auto-downloaded if absent)
python merge_datasets.py --scraped data/politifact_clean.csv

# With local LIAR files
python merge_datasets.py --scraped data/politifact_clean.csv --liar-dir data/liar/

# Temporal split (train < 2020, test >= 2020)
python merge_datasets.py --scraped data/politifact_clean.csv \
                         --temporal-split --split-date 2020-01-01
```

**Output in `data/merged/`:**
- `merged_full.csv` — complete deduplicated dataset
- `merged_train.csv` / `merged_test.csv` — ready for model training
- `merge_stats.txt` — statistics report

---

## 4. Google Fact Check API (bonus)

```bash
# Set your API key
export GOOGLE_FACTCHECK_API_KEY="your_key_here"

# Fetch with default political queries
python google_factcheck.py

# Single query
python google_factcheck.py --query "election fraud 2020"

# Queries from file (one per line)
python google_factcheck.py --queries-file my_queries.txt
```

Then clean and merge as above:

```bash
python clean_and_harmonize.py --input data/google_raw.csv --output data/google_clean.csv
python merge_datasets.py --scraped data/google_clean.csv
```

Get a free API key: https://console.developers.google.com/ → enable "Fact Check Tools API"

---

## Full pipeline (one-liner)

```bash
cd scraping/
python politifact_scraper.py --pages 100 && \
python clean_and_harmonize.py && \
python merge_datasets.py
```

---

## Loading the merged data in the notebook

Replace the LIAR loading cell with:

```python
import pandas as pd

train_df = pd.read_csv("scraping/data/merged/merged_train.csv")
test_df  = pd.read_csv("scraping/data/merged/merged_test.csv")

# label_binary is already computed; re-run text cleaning if text_clean is empty
# (rows from LIAR have text_clean="" — apply preprocess_text from Section 3)
mask = train_df["text_clean"] == ""
train_df.loc[mask, "text_clean"] = train_df.loc[mask, "statement"].apply(preprocess_text)
```
