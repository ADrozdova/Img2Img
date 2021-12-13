from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1,
                 padding=0, pad=False):
        super(ConvBlock, self).__init__()
        self.if_pad = pad
        self.pad = nn.ReflectionPad2d(input_size)
        self.layers = nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(output_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = x
        if self.if_pad:
            out = self.pad(out)
        return self.layers(out)


class DeconvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1,
                 padding=1):
        super(DeconvBlock, self).__init__()
        self.pad = nn.ReflectionPad2d(input_size)
        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(input_size, output_size, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(output_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class ResBlock(nn.Module):
    def __init__(self, n_filter, kernel_size=3, stride=1, padding=0):
        super(ResBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_filter, n_filter, kernel_size, stride, padding),
            nn.InstanceNorm2d(n_filter),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_filter, n_filter, kernel_size, stride, padding),
            nn.InstanceNorm2d(n_filter)
        )

    def forward(self, x):
        return self.layers(x) + x


class DiscrBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=4, stride=2, padding=1, normalize=True):
        super(DiscrBlock, self).__init__()
        layers = [nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, stride=stride, padding=padding)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_filters))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
