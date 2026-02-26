import os
import gzip
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

BASE_URL = "https://xmrec.github.io/data/us/"
RAW_DIR = "data/xmrec/us/raw"
OUTPUT_DIR = "data/xmrec/us"


def discover_links():
    r = requests.get(BASE_URL)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    categories = {}

    for link in soup.find_all("a"):
        href = link.get("href")
        if not href or not href.endswith(".gz"):
            continue

        full_url = urljoin(BASE_URL, href)
        filename = os.path.basename(urlparse(full_url).path)

        # e.g. "ratings_us_Electronics.txt.gz" -> type="ratings", cat="Electronics"
        parts = filename.replace(".txt.gz", "").replace(".json.gz", "").split("_us_")
        if len(parts) != 2:
            continue

        file_type, category = parts[0], parts[1]

        if category not in categories:
            categories[category] = {}
        categories[category][file_type] = full_url

    return categories


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
                # some files have a JSON array per line, others a single dict
                if isinstance(parsed, list):
                    records.extend(parsed)
                elif isinstance(parsed, dict):
                    records.append(parsed)
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(records)


def process_category(category, urls):
    print(f"\n{'='*60}")
    print(f"{category}")
    print(f"{'='*60}")

    cat_dir = os.path.join(RAW_DIR, category)

    local_files = {}
    for file_type, url in urls.items():
        filename = os.path.basename(urlparse(url).path)
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


def main():
    print("discovering XMRec US market datasets...")
    categories = discover_links()
    print(f"found {len(categories)} categories: {', '.join(categories.keys())}\n")

    all_dfs = []
    for category, urls in categories.items():
        df = process_category(category, urls)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        print("\nno data was loaded")
        return

    print(f"\n{'='*60}")
    print("merging all categories...")
    print(f"{'='*60}")

    merged = pd.concat(all_dfs, ignore_index=True)

    print(f"\nfinal dataset: {merged.shape}")
    print(f"  {len(merged):,} interactions")
    print(f"  {merged['user_id'].nunique():,} unique users")
    print(f"  {merged['item_id'].nunique():,} unique items")
    print(f"  {merged['category'].nunique()} categories")
    print(f"  columns: {list(merged.columns)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "xmrec_us_merged.parquet")
    merged.to_parquet(out_path, index=False)
    print(f"\nsaved to {out_path} ({os.path.getsize(out_path) / (1024**2):.1f} MB)")

    preview_path = os.path.join(OUTPUT_DIR, "xmrec_us_preview.csv")
    merged.head(100).to_csv(preview_path, index=False)
    print(f"preview (100 rows) saved to {preview_path}")


if __name__ == "__main__":
    main()