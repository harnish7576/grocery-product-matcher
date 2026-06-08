# Grocery Product Matching Pipeline

A production-style NLP pipeline that matches equivalent grocery products across two retailers (Walmart and Wegmans) without relying on UPC codes. Built as a take-home engineering assignment using 50K+ rows per store. Input data available in this [folder](https://your-drive-link-here). [Pinning data soon!]

---

## How It Works

The core challenge: match products across two stores using only messy text fields (name, brand, size, category). No barcodes. No shared identifiers.

The pipeline solves this in four stages:

```
Raw CSVs
   ↓
Data Cleaning  →  Shared schema: name, brand, size, category, storage
   ↓
Field Embedding  →  Each field embedded separately via sentence-transformers
   ↓
FAISS Search  →  Weighted late-fusion cosine similarity, top-K candidates
   ↓
LLM Reranker  →  GPT-4o-mini confirms or rejects ambiguous matches
   ↓
matches.csv
```

**Why separate field embeddings?**  
Embedding name, brand, size, and category as individual vectors (instead of one combined string) lets you control how much each field influences the match at search time. A name match should outweigh a brand match. A size mismatch should penalize more than a category mismatch. This is the late-fusion approach — fields are combined with explicit weights after retrieval, not before.

**Two-tier matching:**
- `similarity ≥ 0.84` → auto-accepted, no LLM call needed
- `0.80 ≤ similarity < 0.84` → sent to GPT-4o-mini for judgment
- `similarity < 0.80` → auto-rejected

This keeps LLM costs low by only calling the API on genuinely ambiguous cases.

---

## Tech Stack

- `sentence-transformers` — `all-MiniLM-L6-v2` for fast, normalized field embeddings
- `FAISS` — IndexFlatIP (inner product) for cosine similarity search
- `GPT-4o-mini` — LLM reranker for the ambiguous middle tier
- `pandas` — data cleaning and schema normalization
- `concurrent.futures` — parallel LLM calls with rate-limit backoff

---

## Project Structure

```
.
├── main.py                   # Full pipeline: clean → embed → search → match
├── openai_creds.yaml         # OpenAI API key and endpoint config (not committed)
├── products_clean.csv        # Cleaned shared-schema output (generated)
├── embeddings_walmart.npz    # Per-field embeddings for Walmart (generated)
├── embeddings_wegmans.npz    # Per-field embeddings for Wegmans (generated)
└── matches.csv               # Final match output (generated)
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install pandas sentence-transformers faiss-cpu openai pyyaml beautifulsoup4 lxml numpy
```

---

## Configuration

Create `openai_creds.yaml` in the project root:

```yaml
openai:
  api_key: "sk-..."
  endpoint: "https://api.openai.com/v1"
  deployment_name: "gpt-4o-mini"
```

> This file is gitignored. Never commit your API key.

---

## Running the Pipeline

Place your input CSVs in the project root, then:

```bash
python main.py
```

Expected input columns:

| Column       | Description                        |
|--------------|------------------------------------|
| `item_id`    | Unique product identifier          |
| `brand_raw`  | Raw brand name                     |
| `name`       | Product display name               |
| `item_info`  | JSON string with size/category     |
| `tags`       | JSON string with category fallback |

The pipeline runs end-to-end and produces `matches.csv`.

---

## Output Format

`matches.csv` contains one row per confirmed match:

| Column              | Description                            |
|---------------------|----------------------------------------|
| `item_id_walmart`   | Walmart product ID                     |
| `item_id_wegmans`   | Matched Wegmans product ID             |
| `similarity_score`  | Weighted late-fusion cosine similarity |
| `name_walmart`      | Walmart product name                   |
| `name_wegmans`      | Wegmans product name                   |
| `match_type`        | `auto_accepted` or `llm_reranked`      |

---

## Tuning

Key constants to adjust for your dataset:

```python
WEIGHTS = {
    "name"    : 0.40,
    "category": 0.25,
    "size"    : 0.20,
    "brand"   : 0.10,
    "storage" : 0.05,
}

AUTO_ACCEPT_THRESHOLD = 0.84   # raise to reduce false positives
AUTO_REJECT_THRESHOLD = 0.80   # lower to send more to LLM
TOP_K = 5                      # candidates per Walmart product
```

---

## Notes

- The embedding step is the slowest part. On a standard laptop CPU it takes ~10-15 minutes for 100K rows. Run it once — the `.npz` files are reusable.
- To swap in a larger embedding model, change `MODEL_NAME` in `main.py`. `all-mpnet-base-v2` gives better semantic accuracy at roughly 3x the cost.
