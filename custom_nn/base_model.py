import sys
import pathlib as pb
from typing import List, Dict, Union

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

DIR_PATH = pb.Path(__file__).resolve().parent
sys.path.append(str(DIR_PATH))
import custom_layers



class BaseStylishFastMRI(nn.Module):
    def __init__(
        self
        , block_kwargs_list: List[Dict]
        , block_name: str='DataConsistedStylishUNet'
        , iterative_type: str='unrolled'
        , num_iterations: int=1
    ) -> None:
                
        super().__init__()
        
        block_fn = getattr(custom_layers, block_name)
        
        self.rec_blocks = nn.ModuleList([
            block_fn(**kwargs)
            for kwargs in block_kwargs_list
        ])
            
        self.iterative_type = iterative_type
        self.num_iterations = num_iterations
        
    def forward(
        self
        , image: torch.Tensor
        , known_freq: torch.Tensor
        , mask: torch.Tensor
        , texture: torch.Tensor=None
        , is_deterministic: bool=False
    ) -> torch.Tensor:
        
        # In case of unrolled reconstruction, sequentially apply reconstruction blocks.
        # Apply data consistency between recon blocks
        if self.iterative_type == 'unrolled':
            for _ in range(self.num_iterations):
                for block in self.rec_blocks:
                    image = block(image, known_freq, mask, texture, is_deterministic=is_deterministic)
                
        # In case of unrolled reconstruction, apply the same recon block 'self.num_iterations' times.
        # Apply data consistency between recon blocks
        elif self.iterative_type == 'rolled':
            block = self.rec_blocks[0]
            for _ in range(self.num_iterations):
                image = block(image, known_freq, mask, texture, is_deterministic=is_deterministic)

        return image
