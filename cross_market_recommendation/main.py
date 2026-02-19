import os
import gc
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
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

# load the dataset — only the columns we actually need
DATA_PATH = os.path.join("data", "xmrec", "us", "xmrec_us_merged.parquet")

print(f"\nLoading dataset from: {DATA_PATH}")

# only load the columns we need (skip heavy text like reviewText, summary)
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
   
NUM_EPOCHS = 50
HIDDEN_DIM = 48
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4     # L2 regularisation — penalises large weights
PATIENCE = 10           # early stopping — stop if val_loss doesn't improve for this many epochs

loss_fn = torch.nn.BCEWithLogitsLoss()

# split edges into 80% train / 20% validation
all_edges = graph.edge_index.t().cpu().numpy()
train_edges, val_edges = train_test_split(all_edges, test_size=0.2, random_state=42)

train_edge_index = torch.tensor(train_edges.T, dtype=torch.long).to(device)
val_edge_index = torch.tensor(val_edges.T, dtype=torch.long).to(device)

print(f"\n  Training edges   : {train_edge_index.size(1):,}")
print(f"  Validation edges : {val_edge_index.size(1):,}")

model = MyGraphModel(
    in_channels=graph.x.size(1),
    hidden_channels=HIDDEN_DIM,
    out_channels=HIDDEN_DIM,
).to(device)

decoder = EdgeDecoder(hidden_channels=HIDDEN_DIM).to(device)

# Adam optimises both encoder and decoder.
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(decoder.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,  # L2 regularisation to prevent overfitting
)

# TRAINING LOOP with early stopping
best_val_loss = float('inf')
epochs_without_improvement = 0

for epoch in range(NUM_EPOCHS + 1):
    model.train()
    decoder.train()

    # forward pass
    z = model(graph.x, train_edge_index)  # z shape: [num_nodes, 48]

    # positive sampling
    pos_edge_index = train_edge_index

    # negative sampling
    neg_edge_index = negative_sampling(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        num_neg_samples=pos_edge_index.size(1),  # same count of negative edges as positives
    )

    # score the edges
    # concatenate source & destination embeddings and pass them through a 2-layer MLP to get a single logit per edge.
    pos_scores = decoder(z, pos_edge_index)  # [num_pos, 1]
    neg_scores = decoder(z, neg_edge_index)  # [num_neg, 1]

    # build labels: 1 for positive, 0 for negative
    pos_labels = torch.ones(pos_scores.size(0), 1).to(device)
    neg_labels = torch.zeros(neg_scores.size(0), 1).to(device)

    # stack everything into one tensor for the loss function.
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


# VALIDATION
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

    # Validation loss
    val_loss = loss_fn(val_scores, val_labels).item()

    # ROC-AUC
    probs = torch.sigmoid(val_scores).cpu().numpy()
    labels_np = val_labels.cpu().numpy()
    auc = roc_auc_score(labels_np, probs)

    print(f"\n  >> Validation Loss: {val_loss:.4f}, AUC: {auc:.4f}")


# validation summary
print(f"\n{'='*50}")
print("VALIDATION RESULTS")
print(f"{'='*50}")
print(f"  Validation Loss : {val_loss:.4f}")
print(f"  Validation AUC  : {auc:.4f}")

if auc >= 0.9:
    print(" Excellent — model separates positives and negatives very well.")
elif auc >= 0.75:
    print(" Good — solid predictive performance.")
elif auc >= 0.6:
    print(" Fair — better than random, but room for improvement.")
else:
    print(" Poor — model struggles to distinguish real from fake edges.")


# example predictions
print(f"\n{'='*50}")
print("EXAMPLE EDGE PREDICTIONS")
print(f"{'='*50}")

with torch.no_grad():
    z = model(graph.x, train_edge_index)

    test_pairs = [
        (0, num_users),
        (0, num_users + 10),
        (1, num_users + 50),
        (5, num_users + 100),
        (10, num_users + 200),
    ]

    for src, dst in test_pairs:
        if src < graph.num_nodes and dst < graph.num_nodes:
            test_edge = torch.tensor([[src], [dst]], dtype=torch.long).to(device)
            score = torch.sigmoid(decoder(z, test_edge)).item()
            verdict = "Likely connected" if score > 0.5 else "Unlikely connected"
            print(f"  User {src} → Item {dst - num_users}: "
                  f"Score = {score:.4f} ({verdict})")