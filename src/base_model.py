import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from typing import List



# Pseudo code
# ReconBlock can be a U-Net model considered in the previous demo
class ReconBlock(nn.Module):
    def __init__(self) -> None:
        super(ReconBlock, self).__init__()
        
        self.downsample = []
        self.upsample = []
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.downsample(self.upsample(image))


# Pseudo code, very complex in reality
class IterativeModel(nn.Module):
    def __init__(self, recon_blocks: List[nn.Module], data_consistency: nn.Module, num_iterations: int, 
                 iterative_type: str = 'unrolled') -> None:
        """ Class-constructor for the iterative reconstruction model.
        
        Args:
            recon_blocks: collection of recon blocks. Contains a single block in case of 'rolled' iterative type
            data_consistency: mr-specific operation
            num_iterations: number of iterations with a single block in case of 'rolled' iterative type
            iterative_type: either 'rolled' or 'unrolled'
        """
        super(IterativeModel, self).__init__()
        
        self.recon_blocks = recon_blocks
        self.data_consistency = data_consistency
        self.iterative_type = iterative_type
        self.num_iterations = num_iterations
        
    def __forward__(self, image: torch.Tensor) -> torch.Tensor:
        # In case of unrolled reconstruction, sequentially apply reconstruction blocks.
        # Apply data consistency between recon blocks
        if self.iterative_type == 'unrolled':
            for block in self.recon_blocks:
                recon = block(image)
                correction = self.data_consistency(recon)
                image = recon - correction
                
        # In case of unrolled reconstruction, apply the same recon block 'self.num_iterations' times.
        # Apply data consistency between recon blocks
        elif self.iterative_type == 'rolled':
            block = self.recon_blocks[0]
            for _ in range(self.num_iterations):
                recon = block(image)
                correction = self.data_consistency(recon)
                image = recon - correction
