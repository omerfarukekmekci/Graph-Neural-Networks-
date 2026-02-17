import os
import gc
import torch
import numpy as np
import pandas as pd
from torch.nn import Linear
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score


# use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# load the dataset — only the columns we actually need
DATA_PATH = os.path.join("data", "xmrec", "us", "xmrec_us_merged.parquet")

print(f"\nLoading dataset from: {DATA_PATH}")

# only load the columns we need (skip heavy text like reviewText, summary)
cols_to_load = ["user_id", "item_id", "rating", "title", "category"]
df = pd.read_parquet(DATA_PATH, columns=cols_to_load)

# ==============================
# SAMPLE THE DATASET
# ==============================
# The full dataset has 26.7M rows / 11.5M users — way too large.
# The node feature matrix alone would be 11.7M × 384 × 4 bytes ≈ 18 GB.
# MovieLens (which edge_predictor.py uses) has ~100K interactions.
# We sample to keep RAM and training time reasonable.

MAX_INTERACTIONS = 1_000_000  # adjust this based on your RAM (500K → ~1-2 GB)

if len(df) > MAX_INTERACTIONS:
    print(f"\n  Dataset too large ({len(df):,} rows), sampling {MAX_INTERACTIONS:,} rows...")
    df = df.sample(n=MAX_INTERACTIONS, random_state=42).reset_index(drop=True)

# Print Dataset Statistics
print(f"\n{'='*50}")
print("DATASET OVERVIEW")
print(f"{'='*50}")
print(f"  Shape:           {df.shape}")
print(f"  Columns:         {list(df.columns)}")
print(f"  Total rows:      {len(df):,}")
print(f"  Unique users:    {df['user_id'].nunique():,}")
print(f"  Unique items:    {df['item_id'].nunique():,}")

# if the 'category' column exists, show category distribution
if "category" in df.columns:
    print(f"  Categories:      {df['category'].nunique()}")
    print(f"\n  Category distribution:")
    for cat, count in df["category"].value_counts().items():
        print(f"    {cat}: {count:,} interactions")


# build the graph
unique_users = df["user_id"].unique()
unique_items = df["item_id"].unique()
num_users = len(unique_users)
num_items = len(unique_items)

# create dictionaries: string ID -> integer index
# offset items by the number of users
user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
item_to_idx = {iid: i + num_users for i, iid in enumerate(unique_items)}

num_nodes = num_users + num_items
print(f"\n  Mapped {num_users:,} users + {num_items:,} items = {num_nodes:,} nodes")

# build edge_index
src = df["user_id"].map(user_to_idx).values
dst = df["item_id"].map(item_to_idx).values

edge_index = torch.tensor(
    np.array([
        np.concatenate([src, dst]),
        np.concatenate([dst, src]),
    ]),
    dtype=torch.long,
)

del src, dst
gc.collect()

print(f"  Built edge_index: shape {list(edge_index.shape)} ({edge_index.shape[1]:,} directed edges)")

# generate item embeddings from title and category to create meaningful features for the GNN
print("\nGenerating item embeddings with sentence-transformers...")

# build item_df once with all needed columns
item_df = df.drop_duplicates(subset=["item_id"])
text_cols = ["item_id"]
if "title" in item_df.columns:
    text_cols.append("title")
if "category" in item_df.columns:
    text_cols.append("category")
item_df = item_df[text_cols].reset_index(drop=True)

del df
gc.collect()

# combine available text columns into one string per item
def build_text(row):
    parts = []
    if "title" in row and pd.notna(row["title"]):
        parts.append(str(row["title"]))
    if "category" in row and pd.notna(row["category"]):
        parts.append(str(row["category"]))
    return " ".join(parts) if parts else "unknown item"

item_texts = item_df.apply(build_text, axis=1).tolist()

del item_df
gc.collect()

# create the embeddings — use GPU if available for faster encoding
st_model = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))
item_embeddings = st_model.encode(item_texts, show_progress_bar=True, batch_size=256)
item_embeddings = torch.tensor(item_embeddings, dtype=torch.float)

del st_model, item_texts
gc.collect()

feature_dim = item_embeddings.shape[1]
print(f"  Item embeddings: {item_embeddings.shape} (384-dim per item)")

# create the node feature matrix
x = torch.zeros(num_nodes, feature_dim)
x[num_users:] = item_embeddings

del item_embeddings
gc.collect()

# package into a PyG Data object
graph = Data(x=x, edge_index=edge_index).to(device)

del x, edge_index
gc.collect()

print(f"\n{'='*50}")
print("GRAPH SUMMARY")
print(f"{'='*50}")
print(f"  Nodes:          {graph.num_nodes:,}")
print(f"  Edges:          {graph.num_edges:,}")
print(f"  Feature dim:    {feature_dim}")
print(f"  User nodes:     0 .. {num_users-1}")
print(f"  Item nodes:     {num_users} .. {num_nodes-1}")

