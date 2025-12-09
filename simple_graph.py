import torch
from torch.nn import Linear
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data

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


x = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 5.0], [5.0, 7.0]], dtype=torch.float)
edge_index = (
    torch.tensor(
        [[0, 1], [1, 0], [0, 2], [2, 0], [2, 1], [1, 2], [2, 3], [3, 2]],
        dtype=torch.long,
    )
    .t()
    .contiguous()
)

graph = Data(x=x, edge_index=edge_index).to(device)

model = MyGraphModel(
    in_channels=graph.x.size(1),
    hidden_channels=64,
    out_channels=64,
).to(device)

out = model(graph.x, graph.edge_index)

print(graph)
