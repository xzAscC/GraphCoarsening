"""GCN and GraphSAGE encoder models for graph representation learning.

Both models support optional edge weights, which is required for inference
on coarsened graphs that have weighted edges.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv


class GCN(torch.nn.Module):
    """Graph Convolutional Network (Kipf & Welling, 2017).

    Standard 3-layer GCN with ReLU activation and dropout between layers.
    The final layer has no activation and no dropout, producing raw node
    embeddings suitable for downstream tasks like link prediction.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through GCN layers.

        Args:
            x: Node feature matrix of shape (N, in_channels).
            edge_index: Edge indices of shape (2, E).
            edge_weight: Optional edge weights of shape (E,). Used for
                coarsened graphs with weighted edges.

        Returns:
            Node embeddings of shape (N, out_channels).
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_weight)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GraphSAGEModel(torch.nn.Module):
    """GraphSAGE encoder (Hamilton et al., 2017).

    Same interface as GCN but uses SAGEConv layers instead. Supports
    optional edge weights via the mean aggregator variant.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through GraphSAGE layers.

        Args:
            x: Node feature matrix of shape (N, in_channels).
            edge_index: Edge indices of shape (2, E).
            edge_weight: Accepted for API compatibility with GCN but
                not used by SAGEConv (SAGE does not support edge weights).

        Returns:
            Node embeddings of shape (N, out_channels).
        """
        _ = edge_weight  # SAGEConv does not support edge weights
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
