import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm



class SpectralConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.module = spectral_norm(nn.Conv2d(*args, **kwargs))
        
    def forward(self, x):
        return self.module(x)
        
        
class UNet(nn.Module):
    """
    A standard UNet network (with padding in covs).

    Args:
      - num_classes: number of output classes
      - min_channels: minimum number of channels in conv layers
      - max_channels: number of channels in the bottleneck block
      - num_down_blocks: number of blocks which end with downsampling

    The full architecture includes downsampling blocks, a bottleneck block and upsampling blocks

    You also need to account for inputs which size does not divide 2**num_down_blocks:
    interpolate them before feeding into the blocks to the nearest size which divides 2**num_down_blocks,
    and interpolate output logits back to the original shape
    """

    def __init__(
        self
        , num_classes=1
        , min_channels=32
        , max_channels=512
        , num_down_blocks=4
    ):

        assert num_down_blocks > 1
        assert max_channels // (2 ** num_down_blocks) == min_channels

        super().__init__()
        self.num_classes = num_classes
        self.num_down_blocks = num_down_blocks
        self.min_side = 2 ** self.num_down_blocks

        # Initial feature extractor from RGB image
        self.init_block = nn.Conv2d(3, min_channels, kernel_size=1)

        self.encoder_blocks = nn.ModuleList([])
        self.encoder_down_blocks = nn.ModuleList([])
        self.decoder_blocks = nn.ModuleList([])
        self.decoder_up_blocks = nn.ModuleList([])
        for b in range(1, num_down_blocks + 1):
            ## Encoder part
            in_channels = min_channels if b == 1 else out_channels
            out_channels = max_channels // 2 ** (num_down_blocks - b)

            encoder_block = self._construct_block(in_channels)
            self.encoder_blocks.append(encoder_block)

            encoder_down_block = self._construct_down_block(in_channels, out_channels)
            self.encoder_down_blocks.append(encoder_down_block)

            if out_channels != max_channels:
                decoder_block = self._construct_block(
                    out_channels
                    , in_channels=out_channels * 2
                    , prepend_with_dropout=True
                )
            else:
                # Bottleneck
                decoder_block = self._construct_block(out_channels)
            self.decoder_blocks.append(decoder_block)

            decoder_up_block = self._construct_up_block(out_channels, in_channels)
            self.decoder_up_blocks.append(decoder_up_block)

        # NOTE: Pretty cheap operation, we just reverse tensor references in the small list
        self.decoder_blocks = self.decoder_blocks[::-1]
        self.decoder_up_blocks = self.decoder_up_blocks[::-1]

        self.final_block = nn.Sequential(
            self._construct_block(
                min_channels
                , in_channels=min_channels * 2
                , prepend_with_dropout=True
            )
            , nn.Conv2d(min_channels, num_classes, kernel_size=1)
        )

    def forward(self, inputs):
        *_, h, w = inputs.shape
        valid_h = self._find_closest_to(h, divisible_by=self.min_side)
        valid_w = self._find_closest_to(w, divisible_by=self.min_side)
        x = F.interpolate(inputs, size=(valid_h, valid_w), mode='bilinear')

        x = self.init_block(x)

        down_features = []
        for e, ed in zip(self.encoder_blocks, self.encoder_down_blocks):
            x = e(x)
            down_features.append(x.clone())
            x = ed(x)

        for i, (d, du) in enumerate(zip(self.decoder_blocks, self.decoder_up_blocks), -1):
            if i < 0:
                # Bottleneck
                x = d(x)
            else:
                x = torch.cat([x, down_features[-(i+1)]], dim=1)
                x = d(x)
            x = du(x)

        x = torch.cat([x, down_features[0]], dim=1)
        x = self.final_block(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear')
        logits = x

        assert logits.shape == (inputs.shape[0], self.num_classes, inputs.shape[2], inputs.shape[3]), 'Wrong shape of the logits'
        
        return logits

    def _construct_block(self, channels, in_channels=None, prepend_with_dropout=False):
        block_layers = [
            SpectralConv2d(channels, channels, kernel_size=3, padding=1, bias=False)
            , nn.BatchNorm2d(channels)
            , nn.LeakyReLU(inplace=True)
            , SpectralConv2d(channels, channels, kernel_size=3, padding=1, bias=False)
            , nn.BatchNorm2d(channels)
            , nn.LeakyReLU(inplace=True)
        ]

        if in_channels is not None:
            # To fuse concatenated tensor
            block_layers.insert(0, nn.Conv2d(in_channels, channels, kernel_size=1))

        if prepend_with_dropout:
            block_layers.insert(0, nn.Dropout2d(0.5))

        return nn.Sequential(*block_layers)

    def _construct_down_block(self, in_channels, out_channels):
        block_layers = [
            nn.MaxPool2d(2)
            , SpectralConv2d(in_channels, out_channels, kernel_size=1)
        ]
        return nn.Sequential(*block_layers)

    def _construct_up_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    @classmethod
    def _find_closest_to(cls, num, divisible_by):
        if num % divisible_by == 0:
            closest = num
        else:
            lesser = num - (num % divisible_by)
            bigger = (num + divisible_by) - (num % divisible_by)
            closest = lesser if abs(num - lesser) < abs(num - bigger) else bigger
        return closest
