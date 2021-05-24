from collections import namedtuple

import torch


        
EntityKwargs = namedtuple('EntityKwargs', ['entity', 'kwargs'])


def to_two_channel_complex(data: torch.Tensor) -> torch.Tensor:
    """ Change data representation from one channel complex-valued to two channers real valued """
    real = data.real
    imag = data.imag
    result = torch.empty((*data.shape, 2), dtype=torch.float32, device=data.device)
    result[..., 0] = real
    result[..., 1] = imag
    return result


def complex_abs(data: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    """ Convert complex image to a magnitude image (projection from complex to a real plane) """
    assert data.size(-1) == 2
    return (data ** 2).sum(dim=-1, keepdim=keepdim).sqrt()


def image_to_kspace(image: torch.Tensor) -> torch.Tensor:
    """ Convert image to the corresponding k-space using Fourier transforms """
    image_shifted = torch.fft.fftshift(image)
    kspace_shifted = torch.fft.fft2(image_shifted)
    kspace = torch.fft.ifftshift(kspace_shifted)
    return kspace


def kspace_to_image(kspace: torch.Tensor) -> torch.Tensor:
    """ Convert k-space to the corresponding image using Fourier transforms 
        Only for single-coil usage.
    """
    kspace_shifted = torch.fft.ifftshift(kspace)
    image_shifted = torch.fft.ifft2(kspace_shifted)
    image = torch.fft.fftshift(image_shifted).real
    return image


def revert_mask(mask):
    return (mask - 1) * -1


def to_zero_one(tensor: torch.Tensor):
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    tensor = tensor.clamp(0, 1)
    return tensor


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
    rec_kspace = complex_abs(to_two_channel_complex(image_to_kspace(rec_image)))
    return kspace_to_image(mask * rec_kspace - mask * known_kspace)
