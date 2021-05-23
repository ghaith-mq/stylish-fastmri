import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=4,
                      stride=stride,
                      padding=1,
                      bias=False,
                      padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(.2)
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    # guided by: https://www.youtube.com/watch?v=SuddDSqGRzg
    def __init__(self,
                 in_channels=3,
                 inter_channels=[64, 128, 256, 512],
                 ):
        super(Discriminator, self).__init__()
        in_channels_concat = in_channels * 2

        self.first_layer = nn.Sequential(
            nn.Conv2d(
                in_channels_concat,
                inter_channels[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode='reflect'
            ),
            nn.LeakyReLU(.2)
        )

        layers = []
        in_channels_inter = inter_channels[0]

        for out_channels_inter in inter_channels[1:]:
            stride = 2
            if (out_channels_inter == inter_channels[-1]):
                stride = 1

            layers.append(ConvBlock(in_channels_inter,
                                    out_channels_inter,
                                    stride=stride,
                                    ))

            in_channels_inter = out_channels_inter

        layers.append(nn.Conv2d(in_channels=inter_channels[-1],
                                out_channels=1,
                                kernel_size=4,
                                stride=1,
                                padding=1,
                                padding_mode='reflect'
                                ))

        self.layers = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.first_layer(x)
        return self.layers(x)
