import torch
import numpy as np
from torch.nn import Linear
from torch_geometric.datasets import MovieLens
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

device = torch.device("cuda")


class MyGraphModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
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

        edge_feat = torch.cat([z_src, z_dst], dim=1)

        x = self.lin1(edge_feat)
        x = torch.relu(x)
        score = self.lin2(x)

        return score


# movielens dataset
print("Loading MovieLens dataset...")
dataset = MovieLens(root='./data/MovieLens', model_name='all-MiniLM-L6-v2')

data = dataset[0]

movie_features = data['movie'].x
feature_dim = movie_features.size(1)

node_types = data.node_types
movie_type_idx = node_types.index('movie')

# merge user and movie nodes into a single set
graph = data.to_homogeneous().to(device)

graph.x = torch.zeros(graph.num_nodes, feature_dim).to(device)
movie_mask = (graph.node_type == movie_type_idx)
graph.x[movie_mask] = movie_features.to(device)

graph.edge_index = graph.edge_index.to(device)

print(f"Graph loaded: {graph.num_nodes} nodes, {graph.num_edges} edges.")


# ===============================
# 5-fold cross-validation
# ===============================

NUM_FOLDS = 5
NUM_EPOCHS = 100

all_edges = graph.edge_index.t().cpu().numpy()

kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

fold_results = []

loss_fn = torch.nn.BCEWithLogitsLoss()

print(f"\nStarting {NUM_FOLDS}-Fold Cross-Validation...")

for fold, (train_idx, val_idx) in enumerate(kf.split(all_edges)):
    print(f"\n{'='*50}")
    print(f"FOLD {fold + 1}/{NUM_FOLDS}")
    print(f"{'='*50}")
    print(f"  Training edges: {len(train_idx)}, Validation edges: {len(val_idx)}")

    # split edges into training and validation for this fold only, to be changed in the next fold
    train_edge_index = torch.tensor(all_edges[train_idx].T, dtype=torch.long).to(device)
    val_edge_index = torch.tensor(all_edges[val_idx].T, dtype=torch.long).to(device)

    # create a fresh model for each fold
    model = MyGraphModel(
        in_channels=graph.x.size(1),
        hidden_channels=64,
        out_channels=64,
    ).to(device)

    decoder = EdgeDecoder(hidden_channels=64).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(decoder.parameters()), lr=0.01
    )

    # training loop
    for epoch in range(NUM_EPOCHS + 1):
        model.train()
        decoder.train()

        out = model(graph.x, train_edge_index)

        pos_edge_index = train_edge_index

        neg_edge_index = negative_sampling(
            edge_index=graph.edge_index,
            num_nodes=graph.num_nodes,
            num_neg_samples=pos_edge_index.size(1),
        )

        pos_scores = decoder(out, pos_edge_index)
        neg_scores = decoder(out, neg_edge_index)

        pos_labels = torch.ones(pos_scores.size(0), 1).to(device)
        neg_labels = torch.zeros(neg_scores.size(0), 1).to(device)

        scores = torch.cat([pos_scores, neg_scores], dim=0)
        labels = torch.cat([pos_labels, neg_labels], dim=0)

        # Compute loss, backpropagate, and update model
        loss = loss_fn(scores, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}, Train Loss: {loss.item():.4f}")

    # validation
    model.eval()
    decoder.eval()

    with torch.no_grad():
        out = model(graph.x, train_edge_index)

        # validating positive edges
        val_pos_scores = decoder(out, val_edge_index)

        # validating negative edges (we need to do this seperately to avoid biasi)
        val_neg_edge_index = negative_sampling(
            edge_index=graph.edge_index,
            num_nodes=graph.num_nodes,
            num_neg_samples=val_edge_index.size(1),
        )
        val_neg_scores = decoder(out, val_neg_edge_index)

        val_scores = torch.cat([val_pos_scores, val_neg_scores], dim=0)
        val_labels = torch.cat([
            torch.ones(val_pos_scores.size(0), 1),
            torch.zeros(val_neg_scores.size(0), 1),
        ], dim=0).to(device)

        val_loss = loss_fn(val_scores, val_labels).item()

        # area under the ROC curve:
        probs = torch.sigmoid(val_scores).cpu().numpy()
        labels_np = val_labels.cpu().numpy()
        auc = roc_auc_score(labels_np, probs)

        print(f"\n  >> Validation Loss: {val_loss:.4f}, AUC: {auc:.4f}")
        fold_results.append({'loss': val_loss, 'auc': auc})


# =================
# print summary
# =================

print(f"\n{'='*50}")
print("5-FOLD CROSS-VALIDATION RESULTS")
print(f"{'='*50}")
for i, r in enumerate(fold_results):
    print(f"  Fold {i+1}: Loss = {r['loss']:.4f}, AUC = {r['auc']:.4f}")

avg_loss = np.mean([r['loss'] for r in fold_results])
avg_auc = np.mean([r['auc'] for r in fold_results])
std_auc = np.std([r['auc'] for r in fold_results])

print(f"\n  Average Loss: {avg_loss:.4f}")
print(f"  Average AUC:  {avg_auc:.4f} +/- {std_auc:.4f}")


# ====================
# Example Predictions
# ====================

print(f"\n{'='*50}")
print("EXAMPLE EDGE PREDICTIONS (from last fold's model)")
print(f"{'='*50}")

with torch.no_grad():
    out = model(graph.x, train_edge_index)

    test_pairs = [(0, 100), (0, 300), (1, 700), (5, 501), (101, 1000)]
    for src, dst in test_pairs:
        if src < graph.num_nodes and dst < graph.num_nodes:
            test_edge = torch.tensor([[src], [dst]], dtype=torch.long).to(device)
            score = torch.sigmoid(decoder(out, test_edge)).item()
            verdict = "Likely connected" if score > 0.5 else "Unlikely connected"
            print(f"  Node {src} -> Node {dst}: Score = {score:.4f} ({verdict})")