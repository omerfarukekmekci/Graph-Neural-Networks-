"""
Cross-Market Recommendation with LightGCN
==========================================
Train on source market(s), evaluate on target market(s).
Items (ASINs) are shared across markets; users may overlap.
LightGCN learns embeddings from graph structure alone (no text features).

Key design: propagate embeddings through a JOINT graph (source + target edges)
so that target-market users/items get meaningful embeddings via shared items,
but train the BPR loss only on source market edges.
"""

import os, random, glob
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION — change these to pick source / target markets
# ═══════════════════════════════════════════════════════════════

SOURCE_MARKETS = ["au", "ca", "de", "es", "fr", "in", "it", "jp", "mx", "uk"]   # train on these
TARGET_MARKETS = ["ae", "br", "cn", "nl", "sa", "sg", "tr", "us"]   # evaluate on these

MAX_INTERACTIONS = 250_000  # cap per market
EMBED_DIM        = 64
NUM_LAYERS       = 3
EPOCHS           = 50
LR               = 1e-3
WEIGHT_DECAY     = 1e-5
REG_WEIGHT       = 1e-4     # L2 reg on embeddings (standard LightGCN)
PATIENCE         = 10
N_NEG_EVAL       = 99
MAX_EVAL_USERS   = 2000
BATCH_EDGES      = 200_000

DATA_DIR = os.path.join("data", "xmrec")


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_market(market):
    path = os.path.join(DATA_DIR, market, f"xmrec_{market}_merged.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path, columns=["user_id", "item_id", "rating"])
    if len(df) > MAX_INTERACTIONS:
        df = df.sample(n=MAX_INTERACTIONS, random_state=42).reset_index(drop=True)
    return df

# discover available markets if TARGET_MARKETS is empty
available = sorted([
    os.path.basename(d) for d in glob.glob(os.path.join(DATA_DIR, "*"))
    if os.path.isdir(d)
])
if not TARGET_MARKETS:
    TARGET_MARKETS = [m for m in available if m not in SOURCE_MARKETS]

print(f"Source: {SOURCE_MARKETS}  |  Target: {TARGET_MARKETS}")
print(f"Available markets: {available}\n")

# load source
source_dfs = []
for m in SOURCE_MARKETS:
    df = load_market(m)
    if df is not None:
        df["market"] = m
        source_dfs.append(df)
        print(f"  [{m.upper()}] {len(df):,} interactions")
source_df = pd.concat(source_dfs, ignore_index=True)

# load targets
target_data = {}
for m in TARGET_MARKETS:
    df = load_market(m)
    if df is not None:
        target_data[m] = df
        print(f"  [{m.upper()}] {len(df):,} interactions")

print(f"\nSource total: {len(source_df):,} interactions")
print(f"Target markets loaded: {list(target_data.keys())}")


# ═══════════════════════════════════════════════════════════════
# GRAPH CONSTRUCTION (unified ID space)
# ═══════════════════════════════════════════════════════════════
# save source-only sets BEFORE merging, for overlap diagnostics
source_users = set(source_df["user_id"])
source_items = set(source_df["item_id"])

# collect all user/item IDs from source + targets
all_users = set(source_users)
all_items = set(source_items)
for df in target_data.values():
    all_users |= set(df["user_id"])
    all_items |= set(df["item_id"])

user_list = sorted(all_users)
item_list = sorted(all_items)
n_users, n_items = len(user_list), len(item_list)
n_nodes = n_users + n_items

user2idx = {u: i for i, u in enumerate(user_list)}
item2idx = {it: i + n_users for i, it in enumerate(item_list)}

print(f"\nGraph: {n_users:,} users + {n_items:,} items = {n_nodes:,} nodes")

# source edges (bidirectional) — used for BPR loss
src = source_df["user_id"].map(user2idx).values
dst = source_df["item_id"].map(item2idx).values
source_edge_index = torch.tensor(
    np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])]),
    dtype=torch.long,
)
print(f"Source edges: {source_edge_index.shape[1]:,}")

# ═══════════════════════════════════════════════════════════════
# TARGET EDGE SPLITTING: 50% structure (propagation) / 50% eval (held out)
# ═══════════════════════════════════════════════════════════════
# Structure edges go into the joint graph so target users/items
# get meaningful embeddings. Eval edges are NEVER seen by the
# model — used purely for testing. This prevents data leakage.
target_structure_edges = {}  # go into joint graph
target_eval_edges = {}       # held out for evaluation
print(f"\n--- Cross-market overlap diagnostics ---")
for m, df in target_data.items():
    # check overlap with SOURCE (not all)
    t_users = set(df["user_id"])
    t_items = set(df["item_id"])
    user_overlap = t_users & source_users
    item_overlap = t_items & source_items

    # shuffle and split interactions 50/50
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split = len(df_shuffled) // 2
    df_struct = df_shuffled.iloc[:split]
    df_eval   = df_shuffled.iloc[split:]

    # structure edges (bidirectional) — added to joint graph
    ts = df_struct["user_id"].map(user2idx).values
    td = df_struct["item_id"].map(item2idx).values
    te_struct = torch.tensor(
        np.stack([np.concatenate([ts, td]), np.concatenate([td, ts])]),
        dtype=torch.long,
    )
    target_structure_edges[m] = te_struct

    # eval edges (bidirectional) — held out
    es = df_eval["user_id"].map(user2idx).values
    ed = df_eval["item_id"].map(item2idx).values
    te_eval = torch.tensor(
        np.stack([np.concatenate([es, ed]), np.concatenate([ed, es])]),
        dtype=torch.long,
    )
    target_eval_edges[m] = te_eval

    print(f"  [{m.upper()}] struct={te_struct.shape[1]:,}  eval={te_eval.shape[1]:,} | "
          f"user overlap: {len(user_overlap):,}/{len(t_users):,} ({100*len(user_overlap)/len(t_users):.1f}%) | "
          f"item overlap: {len(item_overlap):,}/{len(t_items):,} ({100*len(item_overlap)/len(t_items):.1f}%)")

# ═══════════════════════════════════════════════════════════════
# BUILD JOINT GRAPH for message passing
# ═══════════════════════════════════════════════════════════════
# Joint graph = source edges + target STRUCTURE edges (NOT eval edges)
joint_edges = [source_edge_index] + list(target_structure_edges.values())
joint_edge_index = torch.cat(joint_edges, dim=1)
print(f"\nJoint graph edges (for propagation): {joint_edge_index.shape[1]:,}")

# pick the largest target market as the validation market for early stopping
# use its EVAL edges (not structure edges) to avoid leakage
val_market = max(target_eval_edges, key=lambda m: target_eval_edges[m].shape[1])
val_ei_target = target_eval_edges[val_market]
print(f"Validation market for early stopping: [{val_market.upper()}] ({val_ei_target.shape[1]:,} eval edges)")

del source_df, target_data


# ═══════════════════════════════════════════════════════════════
# LightGCN MODEL
# ═══════════════════════════════════════════════════════════════
class LightGCN(nn.Module):
    """
    LightGCN: Simplifying and Powering Graph Convolution Network
    for Recommendation (He et al., SIGIR 2020).

    - No feature transformation, no activation functions
    - Each layer: E^(k+1) = D^(-1/2) A D^(-1/2) E^(k)
    - Final embedding = mean of all layer outputs
    - Score = dot product
    """
    def __init__(self, n_users, n_items, embed_dim, n_layers):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_emb.weight)

    def _get_ego(self):
        return torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)

    def forward(self, edge_index):
        e0 = self._get_ego()
        # compute symmetric normalisation D^(-1/2)
        row, col = edge_index
        deg = torch.zeros(e0.size(0), device=e0.device)
        deg.scatter_add_(0, row, torch.ones(row.size(0), device=e0.device))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        layers = [e0]
        e = e0
        for _ in range(self.n_layers):
            # sparse message passing: e_new[i] = sum_j norm[i,j] * e[j]
            out = torch.zeros_like(e)
            msg = e[row] * norm.unsqueeze(1)
            out.scatter_add_(0, col.unsqueeze(1).expand_as(msg), msg)
            e = out
            layers.append(e)

        return torch.stack(layers, dim=0).mean(dim=0)  # mean over layers

    def score(self, z, user_idx, item_idx):
        return (z[user_idx] * z[item_idx]).sum(dim=-1)


# ═══════════════════════════════════════════════════════════════
# TRAINING — propagate through JOINT graph, loss on SOURCE edges
# ═══════════════════════════════════════════════════════════════
joint_ei   = joint_edge_index.to(device)
source_ei  = source_edge_index.to(device)
val_ei     = val_ei_target.to(device)

print(f"\nTrain edges (source, for loss): {source_ei.shape[1]:,}")
print(f"Joint edges (for propagation):  {joint_ei.shape[1]:,}")
print(f"Val edges ({val_market.upper()}): {val_ei.shape[1]:,}")

model = LightGCN(n_users, n_items, EMBED_DIM, NUM_LAYERS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

def bpr_loss(z, pos_edge, n_users, n_items):
    """Bayesian Personalised Ranking loss with proper negative item sampling."""
    # only user→item direction (first half of bidirectional edges)
    half = pos_edge.size(1) // 2
    pos_u, pos_i = pos_edge[0, :half], pos_edge[1, :half]
    # sample random ITEMS as negatives (not arbitrary nodes)
    neg_i = torch.randint(n_users, n_users + n_items, (half,), device=pos_edge.device)
    pos_s = (z[pos_u] * z[pos_i]).sum(dim=-1)
    neg_s = (z[pos_u] * z[neg_i]).sum(dim=-1)
    bpr = -torch.log(torch.sigmoid(pos_s - neg_s) + 1e-10).mean()
    # L2 regularization on involved embeddings (standard LightGCN)
    reg = (z[pos_u].pow(2).mean() + z[pos_i].pow(2).mean() + z[neg_i].pow(2).mean()) * REG_WEIGHT
    loss = bpr + reg
    acc  = (pos_s > neg_s).float().mean().item()
    return loss, acc

best_val, patience_ctr = float("inf"), 0
best_state = None

for epoch in range(EPOCHS):
    model.train()

    # subsample SOURCE edges each epoch (like mini-batching for faster convergence)
    if source_ei.size(1) > BATCH_EDGES:
        idx = torch.randperm(source_ei.size(1), device=device)[:BATCH_EDGES]
        batch_ei = source_ei[:, idx]
    else:
        batch_ei = source_ei

    z = model(joint_ei)                                     # propagate over JOINT graph
    loss, train_acc = bpr_loss(z, batch_ei, n_users, n_items)  # loss on SOURCE edges only
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # validation on TARGET market
    model.eval()
    with torch.no_grad():
        z_v = model(joint_ei)
        v_loss, val_acc = bpr_loss(z_v, val_ei, n_users, n_items)
        v_loss = v_loss.item()

    if v_loss < best_val:
        best_val, patience_ctr = v_loss, 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        patience_ctr += 1

    if epoch % 5 == 0:
        print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={loss.item():.4f}  val={v_loss:.4f}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")
    if patience_ctr >= PATIENCE:
        print(f"  Early stop at epoch {epoch}")
        break

model.load_state_dict(best_state)


# ═══════════════════════════════════════════════════════════════
# EVALUATION — proper user→item negative sampling
# ═══════════════════════════════════════════════════════════════
def evaluate(z, edge_index, label, k_values=[10, 20]):
    """Compute AUC and NDCG@K for a set of edges."""
    rng = random.Random(42)
    all_item_idx = list(range(n_users, n_nodes))
    item_set = set(all_item_idx)

    # extract user→item edges only (first half of bidirectional)
    half = edge_index.size(1) // 2
    pos_users = edge_index[0, :half]
    pos_items = edge_index[1, :half]

    # AUC — sample negative ITEMS for each user (not arbitrary nodes)
    neg_items = torch.randint(n_users, n_nodes, (half,))
    pos_u = pos_users.to(device)
    pos_i = pos_items.to(device)
    neg_i = neg_items.to(device)

    pos_s = model.score(z, pos_u, pos_i).cpu()
    neg_s = model.score(z, pos_u, neg_i).cpu()
    scores = torch.cat([pos_s, neg_s]).numpy()
    labels = np.concatenate([np.ones(len(pos_s)), np.zeros(len(neg_s))])
    auc = roc_auc_score(labels, scores)

    # NDCG@K
    user_pos = defaultdict(set)
    s, d = edge_index[0].tolist(), edge_index[1].tolist()
    for u, v in zip(s, d):
        if u < n_users and v >= n_users:
            user_pos[u].add(v)

    eval_users = list(user_pos.keys())
    if len(eval_users) > MAX_EVAL_USERS:
        rng.shuffle(eval_users)
        eval_users = eval_users[:MAX_EVAL_USERS]

    ndcg = {k: 0.0 for k in k_values}
    count = 0
    for u in eval_users:
        pos = list(user_pos[u])
        if not pos:
            continue
        true_item = rng.choice(pos)
        negs = list(item_set - user_pos[u])
        if len(negs) < N_NEG_EVAL:
            continue
        negs = rng.sample(negs, N_NEG_EVAL)
        cands = [true_item] + negs
        u_t = torch.full((len(cands),), u, dtype=torch.long, device=device)
        i_t = torch.tensor(cands, dtype=torch.long, device=device)
        sc = model.score(z, u_t, i_t)
        rank = (sc > sc[0]).sum().item()
        for k in k_values:
            if rank < k:
                ndcg[k] += 1.0 / np.log2(rank + 2)
        count += 1

    ndcg = {k: v / max(count, 1) for k, v in ndcg.items()}
    return auc, ndcg

# run evaluation
model.eval()
with torch.no_grad():
    z = model(joint_ei)

    # source (in-market)
    auc_src, ndcg_src = evaluate(z, source_edge_index, "source")
    print(f"\n{'='*55}")
    print("SOURCE (in-market)")
    print(f"{'='*55}")
    print(f"  AUC: {auc_src:.4f}")
    for k, v in ndcg_src.items():
        print(f"  NDCG@{k}: {v:.4f}")

    # target markets
    print(f"\n{'='*55}")
    print("CROSS-MARKET RESULTS")
    print(f"{'='*55}")
    results = {}
    for m, te in target_eval_edges.items():
        auc_m, ndcg_m = evaluate(z, te, m)
        results[m] = {"auc": auc_m, **{f"ndcg@{k}": v for k, v in ndcg_m.items()}}
        print(f"  [{m.upper()}]  AUC={auc_m:.4f}  " + "  ".join(f"NDCG@{k}={v:.4f}" for k, v in ndcg_m.items()))

# summary table
print(f"\n{'='*55}")
print("SUMMARY")
print(f"{'='*55}")
print(f"  {'Market':<8} {'AUC':>8}  {'NDCG@10':>8}  {'NDCG@20':>8}")
print(f"  {'─'*8} {'─'*8}  {'─'*8}  {'─'*8}")
print(f"  {'SRC':8} {auc_src:8.4f}  {ndcg_src.get(10,0):8.4f}  {ndcg_src.get(20,0):8.4f}")
for m, r in sorted(results.items()):
    print(f"  {m.upper():8} {r['auc']:8.4f}  {r.get('ndcg@10',0):8.4f}  {r.get('ndcg@20',0):8.4f}")

avg_auc = np.mean([r["auc"] for r in results.values()]) if results else 0
print(f"\n  Avg cross-market AUC: {avg_auc:.4f}  (source: {auc_src:.4f}, gap: {auc_src - avg_auc:.4f})")