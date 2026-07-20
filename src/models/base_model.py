

from abc import ABC, abstractmethod
import torch

class BaseAnomalyDetector(ABC, torch.nn.Module):
   
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
                pass

    @abstractmethod
    def compute_anomaly_score(
        self, z: torch.Tensor, src_indices: torch.Tensor, dst_indices: torch.Tensor
    ) -> torch.Tensor:
       
        pass
    