import sys
import pathlib as pb

import torch
import torch.nn as nn

from loguru import logger

DIR_PATH = pb.Path(__file__).resolve().parent
sys.path.append(str(DIR_PATH))
import custom_layers, base_model
# from z_extractor import vanilla_vae, wassersteinae



class StylishFastMRI(base_model.BaseStylishFastMRI):
    
    def __init__(self, z_encoder_kwargs, w_encoder_kwargs, base_model_kwargs):
        super().__init__(**base_model_kwargs)
        
        self.z_encoder = custom_layers.MobileNetV2VAEncoder(**z_encoder_kwargs)
        self.w_encoder = custom_layers.MappingNet(**w_encoder_kwargs)
        
    def forward(self, image: torch.Tensor, known_freq: torch.Tensor, 
                mask: torch.Tensor, texture: torch.Tensor=None, noise: torch.Tensor=None) -> torch.Tensor:
        
        if texture is None:
            z, z_mu, z_log_var = self.z_encoder(image)
            texture = self.w_encoder(z)
        
        out = super().forward(image, known_freq, mask, texture=texture, noise=noise)
        
        logger.debug(f'{out.min()} - {out.max()}, {out.min()} - {out.max()}')
        # logger.debug(f'{z.min()} - {z.max()}, {z.min()} - {z.max()}')
        # logger.debug(f'{texture.min()} - {texture.max()}, {texture.min()} - {texture.max()}')
        
        
        return out, z_mu, z_log_var, texture
