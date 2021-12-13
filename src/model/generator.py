from torch import nn

from src.base.base_model import BaseModel
from src.model.blocks import ConvBlock, ResBlock, DeconvBlock


class Generator(BaseModel):
    def __init__(self, in_size, n_filter, n_blocks, enc_kernels, enc_stride, enc_padding,
                 dec_kernels, dec_stride, dec_padding):
        super(Generator, self).__init__()

        self.enc = nn.Sequential(
            ConvBlock(in_size, n_filter, kernel_size=enc_kernels[0], stride=enc_stride, padding=enc_padding[0],
                      pad=True),
            ConvBlock(n_filter, 2 * n_filter, kernel_size=enc_kernels[1], stride=enc_stride, padding=enc_padding[1]),
            ConvBlock(2 * n_filter, 4 * n_filter, kernel_size=enc_kernels[2], stride=enc_stride, padding=enc_padding[2])
        )
        res_blocks = []
        for i in range(n_blocks):
            res_blocks.append(ResBlock(4 * n_filter))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.dec = nn.Sequential(
            DeconvBlock(4 * n_filter, 2 * n_filter, kernel_size=dec_kernels[0], stride=dec_stride,
                        padding=dec_padding[0]),
            DeconvBlock(2 * n_filter, n_filter, kernel_size=dec_kernels[1], stride=dec_stride, padding=dec_padding[1])
        )
        self.out = nn.Sequential(
            nn.ReflectionPad2d(in_size),
            nn.Conv2d(n_filter, in_size, 7),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.enc(x)
        out = self.res_blocks(out)
        out = self.dec(out)
        return self.out(out)
