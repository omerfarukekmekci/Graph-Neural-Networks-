import os
import gzip
import json
import requests
import pandas as pd

# ── All 18 XMRec markets ──────────────────────────────────────────────────────
MARKETS = [
    "ae", "au", "br", "ca", "cn", "de", "es", "fr",
    "in", "it", "jp", "mx", "nl", "sa", "sg", "tr", "uk", "us",
]

# ── All 16 product categories ─────────────────────────────────────────────────
CATEGORIES = [
    "Arts_Crafts_and_Sewing",
    "Automotive",
    "Books",
    "CDs_and_Vinyl",
    "Cell_Phones_and_Accessories",
    "Digital_Music",
    "Electronics",
    "Grocery_and_Gourmet_Food",
    "Home_and_Kitchen",
    "Industrial_and_Scientific",
    "Kindle_Store",
    "Movies_and_TV",
    "Musical_Instruments",
    "Office_Products",
    "Sports_and_Outdoors",
    "Toys_and_Games",
]

BASE_DOWNLOAD_URL = "https://ciir.cs.umass.edu/downloads/XMarket/FULL"


# ── URL construction (replaces BeautifulSoup scraping) ────────────────────────

def build_urls(market):
    """Build download URLs for every category in a given market.

    Returns dict: {category: {"ratings": url, "reviews": url, "metadata": url}}
    """
    categories = {}
    for cat in CATEGORIES:
        categories[cat] = {
            "ratings":  f"{BASE_DOWNLOAD_URL}/{market}/{cat}/ratings_{market}_{cat}.txt.gz",
            "reviews":  f"{BASE_DOWNLOAD_URL}/{market}/{cat}/reviews_{market}_{cat}.json.gz",
            "metadata": f"{BASE_DOWNLOAD_URL}/{market}/{cat}/metadata_{market}_{cat}.json.gz",
        }
    return categories


# ── Download helper ───────────────────────────────────────────────────────────

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"  [skip] {os.path.basename(dest_path)}")
        return

    print(f"  downloading {os.path.basename(dest_path)}...")
    r = requests.get(url, stream=True)
    r.raise_for_status()

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_ratings(gz_path):
    rows = []
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                rows.append({
                    "user_id": parts[0],
                    "item_id": parts[1],
                    "rating": float(parts[2]),
                    "timestamp": parts[3] if len(parts) > 3 else None,
                })
    return pd.DataFrame(rows)


def load_json_gz(gz_path):
    records = []
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                if isinstance(parsed, list):
                    records.extend(parsed)
                elif isinstance(parsed, dict):
                    records.append(parsed)
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(records)


# ── Per-category processing ───────────────────────────────────────────────────

def process_category(market, category, urls, raw_dir):
    """Download and merge ratings/reviews/metadata for one category."""
    print(f"\n{'='*60}")
    print(f"  [{market.upper()}] {category}")
    print(f"{'='*60}")

    cat_dir = os.path.join(raw_dir, category)

    local_files = {}
    for file_type, url in urls.items():
        filename = os.path.basename(url)
        dest = os.path.join(cat_dir, filename)
        download_file(url, dest)
        local_files[file_type] = dest

    if "ratings" not in local_files:
        print(f"  no ratings file for {category}, skipping")
        return None

    ratings_df = load_ratings(local_files["ratings"])
    ratings_df["category"] = category
    print(f"  {len(ratings_df):,} interactions")

    # merge reviews if available
    if "reviews" in local_files:
        try:
            reviews_df = load_json_gz(local_files["reviews"])
            if not reviews_df.empty:
                review_cols = [c for c in ["asin", "reviewerID", "reviewText", "summary", "overall"]
                               if c in reviews_df.columns]
                reviews_df = reviews_df[review_cols]
                rename_map = {}
                if "asin" in reviews_df.columns:
                    rename_map["asin"] = "item_id"
                if "reviewerID" in reviews_df.columns:
                    rename_map["reviewerID"] = "user_id"
                reviews_df = reviews_df.rename(columns=rename_map)
                ratings_df = ratings_df.merge(reviews_df, on=["user_id", "item_id"], how="left")
                print(f"  {reviews_df.shape[0]:,} reviews merged")
        except Exception as e:
            print(f"  could not load reviews: {e}")

    # merge metadata if available
    if "metadata" in local_files:
        try:
            meta_df = load_json_gz(local_files["metadata"])
            if not meta_df.empty:
                meta_cols = [c for c in ["asin", "title", "price", "averageRating", "categories"]
                             if c in meta_df.columns]
                meta_df = meta_df[meta_cols]
                if "asin" in meta_df.columns:
                    meta_df = meta_df.rename(columns={"asin": "item_id"})
                meta_df = meta_df.drop_duplicates(subset=["item_id"])
                ratings_df = ratings_df.merge(meta_df, on="item_id", how="left")
                print(f"  {meta_df.shape[0]:,} items merged")
        except Exception as e:
            print(f"  could not load metadata: {e}")

    return ratings_df


# ── Per-market processing ─────────────────────────────────────────────────────

def process_market(market):
    """Download, process, and save all categories for one market."""
    output_dir = f"data/xmrec/{market}"
    raw_dir = os.path.join(output_dir, "raw")

    print(f"\n{'#'*60}")
    print(f"  MARKET: {market.upper()}")
    print(f"{'#'*60}")

    categories = build_urls(market)
    print(f"  {len(categories)} categories")

    all_dfs = []
    for category, urls in categories.items():
        df = process_category(market, category, urls, raw_dir)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        print(f"\n  no data loaded for market {market}")
        return

    print(f"\n{'='*60}")
    print(f"  [{market.upper()}] merging all categories...")
    print(f"{'='*60}")

    merged = pd.concat(all_dfs, ignore_index=True)

    print(f"\n  final dataset: {merged.shape}")
    print(f"    {len(merged):,} interactions")
    print(f"    {merged['user_id'].nunique():,} unique users")
    print(f"    {merged['item_id'].nunique():,} unique items")
    print(f"    {merged['category'].nunique()} categories")
    print(f"    columns: {list(merged.columns)}")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"xmrec_{market}_merged.parquet")
    merged.to_parquet(out_path, index=False)
    print(f"\n  saved to {out_path} ({os.path.getsize(out_path) / (1024**2):.1f} MB)")

    preview_path = os.path.join(output_dir, f"xmrec_{market}_preview.csv")
    merged.head(100).to_csv(preview_path, index=False)
    print(f"  preview (100 rows) saved to {preview_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print(f"XMRec dataset fetcher — {len(MARKETS)} markets × {len(CATEGORIES)} categories\n")

    for market in MARKETS:
        process_market(market)

    print(f"\n{'#'*60}")
    print("  ALL DONE")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()