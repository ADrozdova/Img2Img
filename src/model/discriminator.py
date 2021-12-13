from torch import nn

from src.base.base_model import BaseModel
from src.model.blocks import DiscrBlock


class Discriminator(BaseModel):
    def __init__(self, channels, kernel_sz=4, padding=1):
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            DiscrBlock(channels[0], channels[1], kernel_size=kernel_sz, padding=padding, normalize=False),
            DiscrBlock(channels[1], channels[2], kernel_size=kernel_sz, padding=padding),
            DiscrBlock(channels[2], channels[3], kernel_size=kernel_sz, padding=padding),
            DiscrBlock(channels[3], channels[4], kernel_size=kernel_sz, padding=padding),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(channels[4], 1, kernel_size=kernel_sz, padding=padding)
        )

    def forward(self, x):
        return self.layers(x)
