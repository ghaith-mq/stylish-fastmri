import torch
import torch.nn as nn
import torch.nn.functional as F



class NonSaturatingGANLoss(nn.Module):
    def forward(self, fake_scores, real_scores=None):
        if real_scores is None:
            # Generator
            loss = F.binary_cross_entropy_with_logits(fake_scores, torch.ones(fake_scores.shape[0]).to(fake_scores.device))
        else:
            # Discriminator
            loss = F.binary_cross_entropy_with_logits(real_scores, torch.ones(real_scores.shape[0]).to(real_scores.device)) + \
                F.binary_cross_entropy_with_logits(fake_scores, torch.zeros(fake_scores.shape[0]).to(fake_scores.device))
             
        return loss
    
    
class HingeGANLoss(nn.Module):
    def forward(self, fake_scores, real_scores=None):
        if real_scores is None:
            # Generator
            loss = -fake_scores.mean()
        else:
            # Discriminator
            loss = -(torch.min(torch.zeros_like(real_scores), -1 + real_scores) \
                + torch.min(torch.zeros_like(real_scores), -1 - fake_scores)).mean()
             
        return loss
    
    
class KLNormalDivergence(nn.Module):
    def foward(self, mu, log_var):
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
