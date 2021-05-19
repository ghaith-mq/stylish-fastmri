import torch



def to_two_channel_complex(data: torch.Tensor) -> torch.Tensor:
    """ Change data representation from one channel complex-valued to two channers real valued """
    real = data.real
    imag = data.imag
    result = torch.empty((*data.shape, 2), dtype=torch.float32)
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
