import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

device = torch.device("cuda")


class GraphEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()

        self.conv = SAGEConv(in_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = torch.relu(x)
        return x


class EdgeDecoder(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        self.lin1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)

    def forward(self, z, edge_index):
        src, dst = edge_index
        z_src = z[src]
        z_dst = z[dst]

        edge_feat = torch.cat(z_src, z_dst, dim=1)

        intermediate1 = self.lin1(edge_feat)
        intermediate2 = torch.relu(intermediate1)
        score = self.lin2(intermediate2)

        return score


def build_graph():
    x = torch.tensor(
        [
            [1.0, 0.0],  # user 0
            [0.9, 0.1],  # user 1
            [0.1, 0.9],  # item 0
            [0.2, 0.8],  # item 1
            [2.0, 3.0],  # item 2
        ],
        dtype=torch.float,
    )

    edge_index = (
        torch.tensor(
            [[0, 2, 1, 3, 0, 4], [2, 0, 3, 1, 4, 0]],
            dtype=torch.long,
        )
        .t()
        .contiguous()
    )

    graph = Data(x=x, edge_index=edge_index).to(device)

    return graph


def negative_sampling(edge_index, num_nodes, num_negatives):
    existing_edges = set((int(u), int(v)) for u, v in edge_index.t().tolist())

    neg_edges = []

    while len(neg_edges) < num_negatives:
        u = torch.randint(0, num_nodes, (1,)).item()
        v = torch.randint(0, num_nodes, (1,)).item()

        if (u, v) in existing_edges:
            continue

        neg_edges.append([u, v])

    return torch.tensor(neg_edges, dtype=torch.long).t()
