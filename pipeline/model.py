import torch
from torch.nn import Linear
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
