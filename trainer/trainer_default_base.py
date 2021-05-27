import sys
import pathlib as pb
import argparse as ap
import typing as T
import datetime

import tqdm
from loguru import logger

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import piq

ROOT_PATH = pb.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_PATH))
from trainer import trainer_default



class FastMRIDefaultBaseTrainer(trainer_default.FastMRIDefaultTrainer):
            
    def _generator_train_step(self, image, known_freq, known_image, mask, criterion, **kwargs):
        rec_image = self.model(image, known_freq, mask)
        
        cache = {}
        
        loss_rec = criterion.rec(rec_image, known_image)
        cache['loss_rec'] = loss_rec.item()
        
        loss = loss_rec * criterion.rec.coef_
            
        if hasattr(criterion, 'adv'):
            fake_scores = self.discriminator(rec_image, image)
            loss_adv = criterion.adv(fake_scores)
            cache['loss_adv'] = loss_adv.item()
            loss += loss_adv * criterion.adv.coef_
            
        cache['reconstruction'] = rec_image.detach()
        
        return loss, cache
    
    def _generator_val_step(self, image, known_freq, known_image, mask, **kwargs):
        rec_image = self.model(image, known_freq, mask, is_deterministic=True)
        rec_image = rec_image.detach()
        
        cache = {}
        
        cache['metric_ssim'] = piq.ssim(rec_image, known_image, data_range=1.).item()
        cache['metric_psnr'] = piq.psnr(rec_image, known_image, data_range=1.).item()
        cache['reconstruction'] = rec_image
        
        return None, cache
