import os
import gc
import random
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.nn import Linear
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# load the dataset
DATA_PATH = os.path.join("data", "xmrec", "us", "xmrec_us_merged.parquet")

print(f"\nLoading dataset from: {DATA_PATH}")

# only load the columns we need
cols_to_load = ["user_id", "item_id", "rating", "title", "category"]
df = pd.read_parquet(DATA_PATH, columns=cols_to_load)

# sample the dataset
MAX_INTERACTIONS = 2_000_000

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


# Split domains as target and source, to train the model on and test the model on

SOURCE_CATEGORIES = [
    "Books",
    "Electronics",
    "Cell_Phones_and_Accessories",
    "Office_Products",
    "Home_and_Kitchen",
    "Arts_Crafts_and_Sewing",
    "Industrial_and_Scientific",
    "Sports_and_Outdoors",
]

TARGET_CATEGORIES = [
    "Automotive",
    "Grocery_and_Gourmet_Food",
    "Musical_Instruments",
    "Toys_and_Games",
]

source_df = df[df["category"].isin(SOURCE_CATEGORIES)].reset_index(drop=True)
target_df = df[df["category"].isin(TARGET_CATEGORIES)].reset_index(drop=True)

# find overlapping users — users who appear in BOTH source and target domains
# these are the only users we can evaluate cross-domain transfer on,
source_users = set(source_df["user_id"].unique())
target_users = set(target_df["user_id"].unique())
overlapping_users = source_users & target_users

# filter target to only overlapping users since we can only test users the model has seen
target_df = target_df[target_df["user_id"].isin(overlapping_users)].reset_index(drop=True)

print(f"\n{'='*50}")
print("CROSS-DOMAIN SPLIT")
print(f"{'='*50}")
print(f"  Source categories:     {len(SOURCE_CATEGORIES)}")
print(f"  Target categories:     {len(TARGET_CATEGORIES)}")
print(f"  Source interactions:    {len(source_df):,}")
print(f"  Target interactions:   {len(target_df):,} (overlapping users only)")
print(f"  Source users:           {len(source_users):,}")
print(f"  Target users (total):  {len(target_users):,}")
print(f"  Overlapping users:     {len(overlapping_users):,}")


# Build the graph but use only the source nodes for training
unique_users = list(set(
    list(source_df["user_id"].unique()) + list(target_df["user_id"].unique())
))
unique_items = list(set(
    list(source_df["item_id"].unique()) + list(target_df["item_id"].unique())
))
num_users = len(unique_users)
num_items = len(unique_items)

# create dictionaries: string ID -> integer index
# offset items by the number of users
user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
item_to_idx = {iid: i + num_users for i, iid in enumerate(unique_items)}

num_nodes = num_users + num_items
print(f"\n  Mapped {num_users:,} users + {num_items:,} items = {num_nodes:,} nodes")

# build edge_index from source data only
src = source_df["user_id"].map(user_to_idx).values
dst = source_df["item_id"].map(item_to_idx).values

source_edge_index = torch.tensor(
    np.array([
        np.concatenate([src, dst]),
        np.concatenate([dst, src]),
    ]),
    dtype=torch.long,
)

del src, dst
gc.collect()

print(f"  Source edge_index: shape {list(source_edge_index.shape)} ({source_edge_index.shape[1]:,} directed edges)")

# build target edge_index for testing (bidirectional)
target_src = target_df["user_id"].map(user_to_idx).values
target_dst = target_df["item_id"].map(item_to_idx).values

target_edge_index = torch.tensor(
    np.array([
        np.concatenate([target_src, target_dst]),
        np.concatenate([target_dst, target_src]),
    ]),
    dtype=torch.long,
)

del target_src, target_dst
gc.collect()

print(f"  Target edge_index: shape {list(target_edge_index.shape)} ({target_edge_index.shape[1]:,} directed edges)")

# generate item embeddings

print("\nGenerating item embeddings with sentence-transformers...")

# build item_df from both source and target items
all_items_df = pd.concat([source_df, target_df], ignore_index=True)
item_df = all_items_df.drop_duplicates(subset=["item_id"])
text_cols = ["item_id"]
if "title" in item_df.columns:
    text_cols.append("title")
if "category" in item_df.columns:
    text_cols.append("category")
item_df = item_df[text_cols].reset_index(drop=True)

# save domain-specific item node indices before deleting dataframes
# these are needed later for NDCG evaluation (rank within the correct domain)
source_item_indices = [item_to_idx[iid] for iid in source_df["item_id"].unique() if iid in item_to_idx]
target_item_indices = [item_to_idx[iid] for iid in target_df["item_id"].unique() if iid in item_to_idx]

print(f"  Source items for NDCG pool: {len(source_item_indices):,}")
print(f"  Target items for NDCG pool: {len(target_item_indices):,}")

# save per-category edge indices for per-category evaluation later
# each entry maps category name -> edge_index tensor for that category
per_category_edges = {}
for cat in TARGET_CATEGORIES:
    cat_df = target_df[target_df["category"] == cat]
    if len(cat_df) == 0:
        continue
    cat_src = cat_df["user_id"].map(user_to_idx).values
    cat_dst = cat_df["item_id"].map(item_to_idx).values
    cat_edge_index = torch.tensor(
        np.array([
            np.concatenate([cat_src, cat_dst]),
            np.concatenate([cat_dst, cat_src]),
        ]),
        dtype=torch.long,
    )
    per_category_edges[cat] = cat_edge_index
    print(f"    {cat}: {cat_edge_index.shape[1]:,} directed edges")

# also save per-category item indices for NDCG candidate pools
per_category_items = {}
for cat in TARGET_CATEGORIES:
    cat_items = target_df[target_df["category"] == cat]["item_id"].unique()
    per_category_items[cat] = [item_to_idx[iid] for iid in cat_items if iid in item_to_idx]

del df, source_df, target_df, all_items_df
gc.collect()

# combine available text columns into one string per item
def build_text(row):
    parts = []
    if "title" in row and pd.notna(row["title"]):
        parts.append(str(row["title"]))
    if "category" in row and pd.notna(row["category"]):
        parts.append(str(row["category"]))
    return " ".join(parts) if parts else "unknown item"

# build a lookup so embeddings align with the unique_items ordering
item_text_map = {}
for _, row in item_df.iterrows():
    item_text_map[row["item_id"]] = build_text(row)

item_texts = [item_text_map.get(iid, "unknown item") for iid in unique_items]

del item_df, item_text_map
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

# package into a PyG Data object (source edges only)
graph = Data(x=x, edge_index=source_edge_index).to(device)
target_edge_index = target_edge_index.to(device)

del x, source_edge_index
gc.collect()

print(f"\n{'='*50}")
print("GRAPH SUMMARY")
print(f"{'='*50}")
print(f"  Nodes:           {graph.num_nodes:,}")
print(f"  Source edges:     {graph.num_edges:,}")
print(f"  Target edges:    {target_edge_index.shape[1]:,} (held out for cross-domain test)")
print(f"  Feature dim:     {feature_dim}")
print(f"  User nodes:      0 .. {num_users-1}")
print(f"  Item nodes:      {num_users} .. {num_nodes-1}")


# Model definiton
# input -> SAGEConv -> ReLU -> Dropout -> SAGEConv -> output
class MyGraphModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.2):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)  # prevents overfitting
        x = self.conv2(x, edge_index)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(hidden_channels * 2, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z, edge_index):
        src, dst = edge_index
        z_src = z[src]
        z_dst = z[dst]

        # concatenate source and destination embeddings
        edge_feat = torch.cat([z_src, z_dst], dim=1)

        x = self.lin1(edge_feat)
        x = torch.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)  # prevents overfitting
        score = self.lin2(x)  # [num_pairs, 1] — raw logits (not probabilities)

        return score


def compute_ndcg(z, decoder, edge_index, all_item_indices, k_values=[10, 20],
                 n_neg=99, max_users=2000, seed=42, exclude_edges=None):
    # Compute NDCG@K for link prediction using the "1 positive + N negatives" protocol.
    rng = random.Random(seed)
    all_item_set = set(all_item_indices)

    # build per-user positive item sets from the given edges
    user_pos_items = defaultdict(set)
    src_nodes = edge_index[0].cpu().tolist()
    dst_nodes = edge_index[1].cpu().tolist()
    for s, d in zip(src_nodes, dst_nodes):
        # only keep user -> item direction (user indices < num_users)
        if s < num_users and d >= num_users:
            user_pos_items[s].add(d)

    # also build a set of ALL items each user has interacted with (including
    # source-domain items) so we don't sample them as negatives
    user_known_items = defaultdict(set)
    for u in user_pos_items:
        user_known_items[u] = set(user_pos_items[u])

    if exclude_edges is not None:
        ex_src = exclude_edges[0].cpu().tolist()
        ex_dst = exclude_edges[1].cpu().tolist()
        for s, d in zip(ex_src, ex_dst):
            if s < num_users and d >= num_users:
                user_known_items[s].add(d)

    # pick up to max_users for evaluation
    eval_users = list(user_pos_items.keys())
    if len(eval_users) > max_users:
        rng.shuffle(eval_users)
        eval_users = eval_users[:max_users]

    ndcg_sums = {k: 0.0 for k in k_values}
    count = 0

    for user in eval_users:
        pos_items = list(user_pos_items[user])
        if not pos_items:
            continue

        # pick one positive item
        true_item = rng.choice(pos_items)

        # sample n_neg negative items (items the user has NOT interacted with
        # in ANY domain — source or target)
        neg_candidates = list(all_item_set - user_known_items[user])
        if len(neg_candidates) < n_neg:
            continue  # skip users without enough negatives
        neg_items = rng.sample(neg_candidates, n_neg)

        # score all candidate items (1 positive + n_neg negatives)
        candidate_items = [true_item] + neg_items
        user_tensor = torch.tensor([user] * len(candidate_items), dtype=torch.long).to(z.device)
        item_tensor = torch.tensor(candidate_items, dtype=torch.long).to(z.device)
        pair_edge = torch.stack([user_tensor, item_tensor], dim=0)

        scores = decoder(z, pair_edge).squeeze(-1)  # [1 + n_neg]

        # rank by descending score
        _, ranked_indices = torch.sort(scores, descending=True)
        ranked_indices = ranked_indices.cpu().tolist()

        # the true item is at index 0 in candidate_items
        rank_of_true = ranked_indices.index(0)  # 0-indexed rank

        # NDCG: relevance is 1 for the true item, 0 for everything else
        for k in k_values:
            if rank_of_true < k:
                # DCG = 1 / log2(rank + 2), IDCG = 1 / log2(2) = 1.0
                dcg = 1.0 / np.log2(rank_of_true + 2)
                ndcg_sums[k] += dcg / 1.0  # IDCG = 1.0 (best case: true item at rank 0)
            # else: DCG = 0, so NDCG contribution is 0

        count += 1

    if count == 0:
        return {k: 0.0 for k in k_values}

    return {k: ndcg_sums[k] / count for k in k_values}


# ============================================================
# TRAINING SETUP
# ============================================================
NUM_EPOCHS = 30
HIDDEN_DIM = 48
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4     # L2 regularisation — penalises large weights
PATIENCE = 10           # early stopping — stop if val_loss doesn't improve for this many epochs

loss_fn = torch.nn.BCEWithLogitsLoss()

# split SOURCE edges into 80% train / 20% validation
all_edges = graph.edge_index.t().cpu().numpy()
train_edges, val_edges = train_test_split(all_edges, test_size=0.2, random_state=42)

train_edge_index = torch.tensor(train_edges.T, dtype=torch.long).to(device)
val_edge_index = torch.tensor(val_edges.T, dtype=torch.long).to(device)

print(f"\n  Training edges   : {train_edge_index.size(1):,}")
print(f"  Validation edges : {val_edge_index.size(1):,}")
print(f"  Test edges       : {target_edge_index.size(1):,} (cross-domain, unseen categories)")

model = MyGraphModel(
    in_channels=graph.x.size(1),
    hidden_channels=HIDDEN_DIM,
    out_channels=HIDDEN_DIM,
).to(device)

decoder = EdgeDecoder(hidden_channels=HIDDEN_DIM).to(device)

# Adam optimises both encoder and decoder
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(decoder.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,  # L2 regularisation to prevent overfitting
)


# ============================================================
# TRAINING LOOP (with early stopping)
# ============================================================
best_val_loss = float('inf')
epochs_without_improvement = 0

for epoch in range(NUM_EPOCHS + 1):
    model.train()
    decoder.train()

    # forward pass — only source-domain edges
    z = model(graph.x, train_edge_index)  # z shape: [num_nodes, 48]

    # positive sampling
    pos_edge_index = train_edge_index

    # negative sampling
    neg_edge_index = negative_sampling(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        num_neg_samples=pos_edge_index.size(1),  # same count as positives
    )

    # score the edges
    # concatenate source & destination embeddings → 2-layer MLP → single logit per edge
    pos_scores = decoder(z, pos_edge_index)  # [num_pos, 1]
    neg_scores = decoder(z, neg_edge_index)  # [num_neg, 1]

    # build labels: 1 for positive, 0 for negative
    pos_labels = torch.ones(pos_scores.size(0), 1).to(device)
    neg_labels = torch.zeros(neg_scores.size(0), 1).to(device)

    # stack everything into one tensor for the loss function
    scores = torch.cat([pos_scores, neg_scores], dim=0)
    labels = torch.cat([pos_labels, neg_labels], dim=0)

    # compute loss, backpropagate, update weights
    loss = loss_fn(scores, labels)
    optimizer.zero_grad()   # reset gradients from previous step
    loss.backward()         # computes gradients
    optimizer.step()        # updates model weights based on the gradients and learning rate

    # early stopping: check validation loss every epoch
    model.eval()
    decoder.eval()
    with torch.no_grad():
        z_val = model(graph.x, train_edge_index)
        vp_scores = decoder(z_val, val_edge_index)
        vn_edge = negative_sampling(
            edge_index=graph.edge_index,
            num_nodes=graph.num_nodes,
            num_neg_samples=val_edge_index.size(1),
        )
        vn_scores = decoder(z_val, vn_edge)
        v_scores = torch.cat([vp_scores, vn_scores], dim=0)
        v_labels = torch.cat([
            torch.ones(vp_scores.size(0), 1),
            torch.zeros(vn_scores.size(0), 1),
        ], dim=0).to(device)
        epoch_val_loss = loss_fn(v_scores, v_labels).item()

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        epochs_without_improvement = 0
        # save the best model weights
        best_model_state = model.state_dict()
        best_decoder_state = decoder.state_dict()
    else:
        epochs_without_improvement += 1

    if epoch % 5 == 0:
        print(f"  Epoch {epoch:3d}/{NUM_EPOCHS}, Train Loss: {loss.item():.4f}, Val Loss: {epoch_val_loss:.4f}")

    if epochs_without_improvement >= PATIENCE:
        print(f"  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
        break

# restore best model weights
model.load_state_dict(best_model_state)
decoder.load_state_dict(best_decoder_state)


# ============================================================
# SOURCE-DOMAIN VALIDATION (in-domain performance)
# ============================================================
model.eval()
decoder.eval()

with torch.no_grad():
    z = model(graph.x, train_edge_index)

    # score real (positive) validation edges
    val_pos_scores = decoder(z, val_edge_index)

    # score negative validation edges
    val_neg_edge_index = negative_sampling(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        num_neg_samples=val_edge_index.size(1),
    )
    val_neg_scores = decoder(z, val_neg_edge_index)

    val_scores = torch.cat([val_pos_scores, val_neg_scores], dim=0)
    val_labels = torch.cat([
        torch.ones(val_pos_scores.size(0), 1),
        torch.zeros(val_neg_scores.size(0), 1),
    ], dim=0).to(device)

    val_loss = loss_fn(val_scores, val_labels).item()
    val_probs = torch.sigmoid(val_scores).cpu().numpy()
    val_labels_np = val_labels.cpu().numpy()
    val_auc = roc_auc_score(val_labels_np, val_probs)

    # NDCG@K (ranking quality) — rank within source domain items only
    val_ndcg = compute_ndcg(z, decoder, val_edge_index, source_item_indices,
                            exclude_edges=train_edge_index)

print(f"\n{'='*50}")
print("SOURCE-DOMAIN VALIDATION (in-domain)")
print(f"{'='*50}")
print(f"  Validation Loss : {val_loss:.4f}")
print(f"  Validation AUC  : {val_auc:.4f}")
for k, score in val_ndcg.items():
    print(f"  Validation NDCG@{k}: {score:.4f}")

if val_auc >= 0.9:
    print("  Excellent — model separates positives and negatives very well.")
elif val_auc >= 0.75:
    print("  Good — solid in-domain predictive performance.")
elif val_auc >= 0.6:
    print("  Fair — better than random, but room for improvement.")
else:
    print("  Poor — model struggles to distinguish real from fake edges.")


# Cross-domain test (target domain — completely unseen categories)

print(f"\n{'='*50}")
print("CROSS-DOMAIN TEST (unseen categories)")
print(f"{'='*50}")
print(f"  Target categories: {TARGET_CATEGORIES}")
print(f"  Target edges:      {target_edge_index.size(1):,}")

with torch.no_grad():
    # use the trained model with source edges to generate node embeddings
    z = model(graph.x, train_edge_index)

    # score real (positive) target edges
    test_pos_scores = decoder(z, target_edge_index)

    # score negative target edges
    # include both source and target edges so negatives don't accidentally
    # overlap with real target interactions
    test_neg_edge_index = negative_sampling(
        edge_index=torch.cat([graph.edge_index, target_edge_index], dim=1),
        num_nodes=graph.num_nodes,
        num_neg_samples=target_edge_index.size(1),
    )
    test_neg_scores = decoder(z, test_neg_edge_index)

    test_scores = torch.cat([test_pos_scores, test_neg_scores], dim=0)
    test_labels = torch.cat([
        torch.ones(test_pos_scores.size(0), 1),
        torch.zeros(test_neg_scores.size(0), 1),
    ], dim=0).to(device)

    test_loss = loss_fn(test_scores, test_labels).item()
    test_probs = torch.sigmoid(test_scores).cpu().numpy()
    test_labels_np = test_labels.cpu().numpy()
    test_auc = roc_auc_score(test_labels_np, test_probs)

    # NDCG@K for cross-domain — rank within target domain items only
    # this is the standard cross-domain protocol: can the model rank the
    # correct target item above other target items it has never trained on?
    test_ndcg = compute_ndcg(z, decoder, target_edge_index, target_item_indices,
                             exclude_edges=graph.edge_index)

print(f"  Test Loss        : {test_loss:.4f}")
print(f"  Test AUC         : {test_auc:.4f}")
for k, score in test_ndcg.items():
    print(f"  Test NDCG@{k}     : {score:.4f}")


# per category breakdown

print(f"\n{'='*50}")
print("PER-CATEGORY CROSS-DOMAIN RESULTS")
print(f"{'='*50}")

per_cat_results = {}

with torch.no_grad():
    z = model(graph.x, train_edge_index)

    for cat, cat_edges in per_category_edges.items():
        cat_edges = cat_edges.to(device)
        n_edges = cat_edges.size(1)

        # AUC for this category
        cat_pos_scores = decoder(z, cat_edges)
        cat_neg_edge = negative_sampling(
            edge_index=torch.cat([graph.edge_index, target_edge_index], dim=1),
            num_nodes=graph.num_nodes,
            num_neg_samples=n_edges,
        )
        cat_neg_scores = decoder(z, cat_neg_edge)

        cat_scores = torch.cat([cat_pos_scores, cat_neg_scores], dim=0)
        cat_labels = torch.cat([
            torch.ones(cat_pos_scores.size(0), 1),
            torch.zeros(cat_neg_scores.size(0), 1),
        ], dim=0).to(device)

        cat_probs = torch.sigmoid(cat_scores).cpu().numpy()
        cat_labels_np = cat_labels.cpu().numpy()
        cat_auc = roc_auc_score(cat_labels_np, cat_probs)

        # NDCG for this category (rank within that category's items)
        cat_items = per_category_items.get(cat, target_item_indices)
        cat_ndcg = compute_ndcg(z, decoder, cat_edges, cat_items,
                                exclude_edges=graph.edge_index)

        per_cat_results[cat] = {"auc": cat_auc, "ndcg": cat_ndcg, "edges": n_edges}

        print(f"\n  {cat}:")
        print(f"    Edges:    {n_edges:,}")
        print(f"    AUC:      {cat_auc:.4f}")
        for k, score in cat_ndcg.items():
            print(f"    NDCG@{k}:  {score:.4f}")


# cross domain transfer summary

print(f"\n{'='*50}")
print("CROSS-DOMAIN TRANSFER SUMMARY")
print(f"{'='*50}")
print(f"  In-domain AUC    : {val_auc:.4f}")
print(f"  Cross-domain AUC : {test_auc:.4f}")
print(f"  AUC drop         : {val_auc - test_auc:.4f}")
for k in val_ndcg:
    print(f"  NDCG@{k} drop     : {val_ndcg[k] - test_ndcg[k]:.4f}  (in-domain {val_ndcg[k]:.4f} → cross-domain {test_ndcg[k]:.4f})")

if test_auc >= 0.8:
    print("\n  Excellent cross-domain transfer!")
    print("  The model generalises well to unseen categories.")
elif test_auc >= 0.65:
    print("\n  Good cross-domain transfer.")
    print("  The model shows meaningful generalisation to unseen categories.")
elif test_auc >= 0.55:
    print("\n  Moderate cross-domain transfer.")
    print("  The model transfers some knowledge, but there is room for improvement.")
else:
    print("\n  Weak cross-domain transfer.")
    print("  The model struggles to generalise beyond its training categories.")
    print("  Consider: more source domains, richer features, or domain adaptation techniques.")