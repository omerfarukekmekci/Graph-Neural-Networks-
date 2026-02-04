import torch
from torch.nn import Linear
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

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


x = torch.tensor([[1.0, 2.0], [7.0, 11.0], [3.0, 5.0], [15.0, 23.0], [31.0, 47.0]], dtype=torch.float)
edge_index = (
    torch.tensor(
        [[0, 2, 1, 2, 1, 3], [2, 0, 2, 1, 3, 1]],
        dtype=torch.long,
    )
    .contiguous()
)

graph = Data(x=x, edge_index=edge_index).to(device)

model = MyGraphModel(
    in_channels=graph.x.size(1),
    hidden_channels=64,
    out_channels=64,
).to(device)

decoder = EdgeDecoder(hidden_channels=64).to(device)

optimizer = torch.optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=0.01)

# training loop
print("Starting training...")
for epoch in range(101): # Run 100 times
    # forward
    out = model(graph.x, graph.edge_index)

    # positive edges (existing ones)
    pos_edge_index = graph.edge_index
    
    # negative edges (non-existing ones)
    neg_edge_index = negative_sampling(
        edge_index=graph.edge_index, num_nodes=graph.num_nodes, num_neg_samples=pos_edge_index.size(1)
    )

    pos_scores = decoder(out, pos_edge_index)
    neg_scores = decoder(out, neg_edge_index)

# 1 for real edges, 0 for fake
    pos_labels = torch.ones(pos_scores.size(0), 1).to(device)
    neg_labels = torch.zeros(neg_scores.size(0), 1).to(device)

    scores = torch.cat([pos_scores, neg_scores], dim=0)
    labels = torch.cat([pos_labels, neg_labels], dim=0)

    # loss
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss = loss_fn(scores, labels)

    # backward
    optimizer.zero_grad() 
    loss.backward()       
    optimizer.step()      

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


test_edge = torch.tensor([[3], [4]], dtype=torch.long).to(device)
prediction = decoder(out, test_edge)
print(f"Score for edge 3->4: {prediction.item()}")