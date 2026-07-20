# src/models/graph_detector.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from src.models.base_model import BaseAnomalyDetector

class BGPGraphSAGEEncoder(BaseAnomalyDetector):
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        
        # Layer 1: Aggregates features from 1-hop local neighbor ASNs
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        
        # Layer 2: Aggregates features from 2-hop structural paths
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr='mean')
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        
        # First aggregation hop + non-linear projection
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        # Second aggregation hop to compile neighborhood context
        x = self.conv2(x, edge_index)
        return x

    def compute_anomaly_score(
        self, z: torch.Tensor, src_indices: torch.Tensor, dst_indices: torch.Tensor
    ) -> torch.Tensor:
        
        # Extract targeted spatial embeddings for edge nodes
        edge_src_embeddings = z[src_indices]
        edge_dst_embeddings = z[dst_indices]
        
        # Vectorized dot product calculation
        scores = torch.sum(edge_src_embeddings * edge_dst_embeddings, dim=-1)
        
        # Maps raw scalar values into probability spaces [0.0 - 1.0]
        # Low similarity dot product maps to a HIGH anomaly score (close to 1.0)
        anomaly_probability = 1.0 - torch.sigmoid(scores)
        return anomaly_probability