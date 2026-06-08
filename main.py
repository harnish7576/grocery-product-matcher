'''
============ ROADMAP FOLLOWED =============
Data cleaning → Text_for_embedding
       ↓
embedding model [Embed fields separately (name, brand, size, category, storage)]
       ↓
vector store (FAISS or similar) [Weighted vector combination per product] [cosine similarity]
       ↓
vector search 
       ↓
LLM reranker → top-K candidates per product(based on similarity scores)
       ↓
Final match / No-match decision
'''

import pandas as pd
import json
import re
from bs4 import BeautifulSoup
import time
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import yaml
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ======================= CLEANING ===========================

# --------------- Helper Functions ----------------
def strip_html(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()

    if text.count("<") > 0 and text.count(">") > 0:
        return BeautifulSoup(text, "lxml").get_text(" ")
    return text

def norm(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()

def handle_bad_line(line):
    print("Skipping bad line:", line)
    print()
    return None 

def safe_json(val):
    if not isinstance(val, str) or not val.strip():
        return {}
    try:
        return json.loads(val)
    except Exception:
        return {}

# ---------- Entity Recognition Functions ------------
STRONG_UNITS = re.compile(
    r"""(\d+(?:\.\d+)?)\s*
    (fl\.?\s*oz|oz|lb|lbs|g|kg|ml|l(?:iter|itre)?|
     qt|quart|pint|pt|gal|gallon|tsp|tbsp|
     ounce|ounces|pound|pounds|gram|grams)
    \b""",
    re.IGNORECASE | re.VERBOSE,
)

WEAK_UNITS = re.compile(
    r"""(\d+(?:\.\d+)?)\s*(count|ct|pack|pk|piece|pieces)\b""",
    re.IGNORECASE,
)

def extract_size(item_info_str, name=""):
    # Priority 1 => explicit clean field
    try:
        explicit = json.loads(item_info_str or "{}").get("size_user_friendly") or ""
    except Exception:
        explicit = ""
    if explicit.strip():
        return explicit.strip()
 
    name = str(name or "")
 
    # Priority 2 => strong units (oz, ml, lb, tsp, etc.) — take last match
    # to avoid "(2 pack) ... 60 Count" grabbing "2 pack" instead of "60 Count"
    strong = STRONG_UNITS.findall(name)
    if strong:
        num, unit = strong[-1]
        return f"{num} {unit}"
 
    # Priority 3 => count/pack as fallback
    weak = WEAK_UNITS.findall(name)
    if weak:
        num, unit = weak[-1]
        return f"{num} {unit}"

    return ""
 
def extract_category(item_info_str, tags_str):
    for source in [item_info_str, tags_str]:
        d = safe_json(source)
        parts = [d.get(f"category_{i}") for i in range(4)]
        path = " > ".join(p for p in parts if p)
        if path:
            return path
    return ""

def extract_storage(name, category):
    text = (str(name or "") + " " + str(category or "")).lower()
    if "frozen" in text: return "Frozen"
    if "refrigerat" in text: return "Refrigerated"
    if "fresh" in text: return "Fresh"
    return "Dry"

# not used at the moment, can be used if needed
PRIVATE_LABELS = {
    "walmart": {"great value", "mainstays", "better homes & gardens",
                "equate", "sam's choice"},
    "wegmans": {"wegmans"},
}
def is_private_label(brand, store):
    return brand.lower() in PRIVATE_LABELS.get(store, set())

# Use this function if dont want Embed fields separately (name, brand, size, category, storage)
def build_text_for_embeddings(row):
    parts = []
    brand, name = norm(row["brand"]), norm(row["name"])
    parts.append(f"{brand} {name}." if brand and brand.lower() not in name.lower() else f"{name}.")
    if row["size"]:      parts.append(f"Size: {row['size']}.")
    if row["category"]:  parts.append(f"Category: {row['category']}.")
    if row["storage"]:   parts.append(f"Storage: {row['storage']}.")
    desc = norm(strip_html(row.get("description", "")))[:100]
    if desc: parts.append(desc)
    return " ".join(parts)

# -------------- Cleaning Function ----------------
def clean(df: pd.DataFrame, store: str) -> pd.DataFrame:
    df = df.copy()
    print(f"\nCleaning {store} products...")
    df["store"]    = store

    print("Identifying brands...")
    df["brand"]    = df["brand_raw"].fillna("").apply(norm)

    print("Extracting product sizes...")
    df["size"] = df.apply( lambda r: extract_size(r.get("item_info", ""), r.get("name", "")), axis=1 )

    print("Searching categories...")
    df["category"] = df.apply( lambda r: extract_category(r.get("item_info",""), r.get("tags","")), axis=1 )

    print("Classifying storage types...")
    df["storage"]  = df.apply( lambda r: extract_storage(r.get("name",""), r["category"]), axis=1 )

    print("Detecting private labels...")
    df["is_private_label"] = df.apply( lambda r: is_private_label(r["brand"], store), axis=1 )

    # df["text_for_embeddings"] = df.apply(build_text_for_embeddings, axis=1)

    return df[[ "item_id", "store", "brand", "name", "size", "category", "storage", "is_private_label"]]


# ========================= EMBEDDING ==============================
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Using MiniLM => faster, light weight embeddings
# Tradeoff => efficiency over embedding quality
# Can switch to a larger model for better performance and semantic accuracy
FIELDS = ["name", "brand", "size", "category", "storage"]
BATCH_SIZE = 400

# Embedding fields separately (name, brand, size, category, storage), assigning weight later on
def embed_field(model, series: pd.Series) -> np.ndarray:
    texts = series.fillna("").astype(str).tolist()
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   
    )
    return vectors.astype(np.float32)

def embed_store(model, df: pd.DataFrame, store: str, output_path: str):
    subset = df[df["store"] == store].reset_index(drop=True)
    print(f"\n── Embedding {store} ({len(subset)} rows) ──")
 
    arrays = {"ids": subset["item_id"].values}
    for field in FIELDS:
        print(f"  Field: {field}")
        arrays[field] = embed_field(model, subset[field])
 
    np.savez(output_path, **arrays)
    print(f"  Saved → {output_path}")
 
    # Jsut a sanity check
    with np.load(output_path, allow_pickle=True) as data:
        for key in data.files:
            print(f"    {key:12s}  shape={data[key].shape}")

def produce_embeddings(df: pd.DataFrame, walmart_out:  str , wegmans_out:  str):
    print(f"\n  Total rows : {len(df)}")
    print(f"  Walmart    : {(df['store'] == 'walmart').sum()}")
    print(f"  Wegmans    : {(df['store'] == 'wegmans').sum()}")

    print(f"\nLoading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME) 

    embed_store(model, df, "walmart", walmart_out)
    embed_store(model, df, "wegmans", wegmans_out)

    print(f"\nCompleted Embedding!")


# =============== VECTOR STORE + SEARCH + LLM RERANKER ====================
# embedding fields separately gives flexibility to add new fields
# and prioritise some fields over others to match
WEIGHTS = {
    "name"    : 0.40, #0.42
    "category": 0.25, #0.20
    "size"    : 0.20, #0.18
    "brand"   : 0.10, #0.15
    "storage" : 0.05, #0.05
}
 
FIELDS = list(WEIGHTS.keys())

def load_embeddings(path: str):
    data = np.load(path, allow_pickle=True)
    ids = data["ids"]

    combined = np.zeros_like(data["name"])
    for field, weight in WEIGHTS.items():
        combined += weight * data[field]
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    combined = combined / np.where(norms == 0, 1, norms)

    field_vecs = {field: data[field] for field in WEIGHTS}

    return ids, combined.astype(np.float32), field_vecs

# FAISS => Minimumm cosine similarity to accept a match
def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim) #faiss.IndexFlatL2(dim)? Euclidean
    index.add(vectors)
    return index

def load_name_lookup(clean_csv: str) -> tuple[dict, dict]:
    df = pd.read_csv(clean_csv, dtype=str)
    walmart_names = df[df["store"] == "walmart"].set_index("item_id")["name"].to_dict()
    wegmans_names = df[df["store"] == "wegmans"].set_index("item_id")["name"].to_dict()
    return walmart_names, wegmans_names

AUTO_ACCEPT_THRESHOLD = 0.84
AUTO_REJECT_THRESHOLD = 0.80
TOP_K = 5

SYSTEM_PROMPT = """You are a grocery product matching assistant.
Given a Walmart product and up to 10 Wegmans candidates, decide which candidate
is the best match — meaning a customer would consider them essentially the same product.
Rules:
- Match on product type, size, and key attributes. Brand differences are acceptable for private label products.
- If NO candidate is a reasonable match, return -1.
- Respond ONLY with valid JSON. No explanation, no markdown.
Response format: {"best_match_index": <0-9 or -1>, "reason": "<one sentence>"}"""

def call_llm(client, model, walmart_name, candidates):
    lines = [f"Walmart product: {walmart_name}\n", "Wegmans candidates:"]
    for i, c in enumerate(candidates):
        lines.append(f"  [{i}] {c['name']} (similarity: {c['score']:.3f})")
    prompt = "\n".join(lines)

    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0,
                max_completion_tokens=100,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            if "429" in str(e):
                wait = 10 * (attempt + 1)  # 10s, 20s, 30s, 40s, 50s
                print(f"    Rate limited — waiting {wait}s...")
                time.sleep(wait)
            else:
                if attempt == 4:
                    print(f"    LLM failed: {e}")
                    return {"best_match_index": -1, "reason": "LLM error"}
                time.sleep(2 ** attempt)

    return {"best_match_index": -1, "reason": "max retries exceeded"}

def late_fusion_score(walmart_field_vecs, wegmans_field_vecs, w_idx, g_idx):
    return sum(
        WEIGHTS[f] * float(np.dot(walmart_field_vecs[f][w_idx], wegmans_field_vecs[f][g_idx]))
        for f in WEIGHTS
    )

def search(
    walmart_embeddings_path: str,
    wegmans_embeddings_path: str,
    clean_csv: str = "products_clean.csv",
    output_path: str = "matches.csv",
):
    # Load embeddings
    print("Loading Walmart embeddings...")
    walmart_ids, walmart_vecs, walmart_field_vecs = load_embeddings(walmart_embeddings_path)
    print("Loading Wegmans embeddings...")
    wegmans_ids, wegmans_vecs, wegmans_field_vecs = load_embeddings(wegmans_embeddings_path)

    # Load names and config
    walmart_names, wegmans_names = load_name_lookup(clean_csv)

    # Initializing OpenAI client
    cfg = yaml.safe_load(open("openai_creds.yaml"))["openai"]
    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["endpoint"])
    model  = cfg["deployment_name"]

    print("Building FAISS index...")
    index = build_faiss_index(wegmans_vecs)

    print(f"Searching top-{TOP_K} candidates per Walmart product (early fusion retrieval, late fusion scoring)...")
    D, I = index.search(walmart_vecs, k=TOP_K)

    auto_accepted, to_rerank, auto_rejected = [], [], 0

    for i in range(len(walmart_ids)):
        top_score = late_fusion_score(walmart_field_vecs, wegmans_field_vecs, i, I[i, 0])
        w_id      = walmart_ids[i]
        w_name    = walmart_names.get(w_id, "")

        # high confidence => skip LLM, accept directly
        if top_score >= AUTO_ACCEPT_THRESHOLD:
            g_id = wegmans_ids[I[i, 0]]
            auto_accepted.append({
                "item_id_A"  : w_id,
                "item_id_B"  : g_id,
                # "similarity_score" : round(float(top_score), 4),
                # "name_walmart"     : w_name,
                # "name_wegmans"     : wegmans_names.get(g_id, ""),
                # "match_type"       : "auto_accepted",
            })
        # ambiguous => queue for LLM reranking
        elif top_score >= AUTO_REJECT_THRESHOLD:
            candidates = [
                {
                    "id"    : wegmans_ids[I[i, j]],
                    "name"  : wegmans_names.get(wegmans_ids[I[i, j]], ""),
                    "score" : late_fusion_score(walmart_field_vecs, wegmans_field_vecs, i, I[i, j])
                }
                for j in range(TOP_K)
                if late_fusion_score(walmart_field_vecs, wegmans_field_vecs, i, I[i, j]) >= AUTO_REJECT_THRESHOLD
            ]
            if candidates:
                to_rerank.append({"w_id": w_id, "w_name": w_name, "candidates": candidates})
            else:
                auto_rejected += 1
        # low confidence — reject outright
        else:
            auto_rejected += 1

    print(f"\nTier breakdown:")
    print(f"  Auto-accepted : {len(auto_accepted)}")
    print(f"  Sent to LLM   : {len(to_rerank)}")
    print(f"  Auto-rejected : {auto_rejected}")

    # LLM reranking
    llm_accepted, llm_rejected = [], 0

    def rerank_one(item):
        result = call_llm(client, model, item["w_name"], item["candidates"])
        return item, result

    print(f"\nReranking {len(to_rerank)} products with LLM...")
    completed = 0
    NUM_WORKERS = 2  # Adjusting based on system and rate limits
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(rerank_one, item): item for item in to_rerank}
        for future in as_completed(futures):
            item, result = future.result()
            best = result.get("best_match_index", -1)
            if best != -1 and 0 <= best < len(item["candidates"]):
                chosen = item["candidates"][best]
                llm_accepted.append({
                    "item_id_A"  : item["w_id"],
                    "item_id_B"  : chosen["id"],
                    # "similarity_score" : round(chosen["score"], 4),
                    # "name_walmart"     : item["w_name"],
                    # "name_wegmans"     : chosen["name"],
                    # "match_type"       : "llm_reranked",
                })
            else:
                llm_rejected += 1
            completed += 1
            if completed % 500 == 0:
                print(f"  Progress: {completed}/{len(to_rerank)}")

    all_matches = auto_accepted + llm_accepted
    df_matches  = pd.DataFrame(all_matches)
    df_matches.to_csv(output_path, index=False)

    print(f"\n── Final Results ──────────────────────────")
    print(f"  Auto-accepted : {len(auto_accepted)}")
    print(f"  LLM-confirmed : {len(llm_accepted)}")
    print(f"  LLM-rejected  : {llm_rejected}")
    print(f"  Total matches : {len(all_matches)}")
    print(f"  Saved → {output_path}")
    return df_matches


def main():
    df_walmart = pd.read_csv('grocery_store_a_items_final.csv', dtype=str, engine='python', on_bad_lines=handle_bad_line)
    df_wegmans = pd.read_csv('grocery_store_b_items_final.csv', dtype=str, engine='python', on_bad_lines=handle_bad_line)

    clean_walmart = clean(df_walmart, "walmart")
    clean_wegmans = clean(df_wegmans, "wegmans")
    df_clean = pd.concat([clean_walmart, clean_wegmans], ignore_index=True)

    df_clean.to_csv('products_clean.csv', index=False)

    produce_embeddings(df_clean, "embeddings_walmart.npz", "embeddings_wegmans.npz")

    search("embeddings_walmart.npz", "embeddings_wegmans.npz", "products_clean.csv", "matches.csv")

main()