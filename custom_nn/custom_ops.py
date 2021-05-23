import sys
import pathlib as pb

import torch

DIR_PATH = pb.Path(__file__).resolve().parent
sys.path.append(str(DIR_PATH))
import utils



def soft_thresholding(u, lambd):
    """https://arxiv.org/pdf/2004.07339.pdf

    Args:
        u: torch.Tensor
        lambd: soft theshold
    """
    u_abs = u.abs()
    return torch.maximum(u_abs - lambd, 0) * (u / u_abs)


def data_consistency(rec_image, known_kspace, mask):
    """https://arxiv.org/pdf/2004.07339.pdf"""
    rec_kspace = utils.complex_abs(utils.to_two_channel_complex(utils.image_to_kspace(rec_image)))
    return utils.kspace_to_image(mask * rec_kspace - mask * known_kspace)
