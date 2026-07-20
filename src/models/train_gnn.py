

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple
from .graph_detector import BGPGraphSAGEEncoder

class BGPTopologyTrainer:
    """
    Handles self-supervised link-prediction training for the GraphSAGE model.
    Teaches the model to recognize valid network peering structures.
    """
    def __init__(self, model: BGPGraphSAGEEncoder, lr: float = 0.01, weight_decay: float = 5e-4):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.BCEWithLogitsLoss()

    def generate_negative_edges(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Generates random negative edges (links that do not exist in the real topology)
        to act as negative training samples.
        """
        num_edges = edge_index.size(1)
        neg_sources = []
        neg_targets = []
        
        existing_edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))

        while len(neg_sources) < num_edges:
            u = torch.randint(0, num_nodes, (1,)).item()
            v = torch.randint(0, num_nodes, (1,)).item()
            
            if u != v and (u, v) not in existing_edges and (v, u) not in existing_edges:
                neg_sources.append(u)
                neg_targets.append(v)
                
        return torch.tensor([neg_sources, neg_targets], dtype=torch.long)

    def train_epoch(self, x: torch.Tensor, pos_edge_index: torch.Tensor, num_nodes: int) -> float:
        """Runs a single optimization pass over the network topology."""
        self.model.train()
        self.optimizer.zero_grad()

        # 1. Forward pass: generate node representations
        node_embeddings = self.model(x, pos_edge_index)

        # 2. Generate negative samples for contrastive learning
        neg_edge_index = self.generate_negative_edges(pos_edge_index, num_nodes)

        # 3. Calculate link logits for positive edges (should be close to 1)
        pos_src, pos_tgt = pos_edge_index[0], pos_edge_index[1]
        pos_logits = torch.sum(node_embeddings[pos_src] * node_embeddings[pos_tgt], dim=-1)

        # 4. Calculate link logits for negative edges (should be close to 0)
        neg_src, neg_tgt = neg_edge_index[0], neg_edge_index[1]
        neg_logits = torch.sum(node_embeddings[neg_src] * node_embeddings[neg_tgt], dim=-1)

        # 5. Compute Binary Cross-Entropy loss
        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.cat([torch.ones(pos_logits.size(0)), torch.zeros(neg_logits.size(0))])

        loss = self.criterion(logits, labels)
        loss.backward()
        self.optimizer.step()

        return loss.item()