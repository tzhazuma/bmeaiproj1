import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_ch, out_ch),
        )

    def forward(self, x):
        return self.mpconv(x)


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(skip_ch + out_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = nn.functional.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                                    diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32, depth=4):
        super().__init__()
        self.depth = depth

        self.inc = ConvBlock(in_channels, base_channels)
        encoder_channels = [base_channels]
        ch = base_channels

        self.downs = nn.ModuleList()
        for _ in range(depth):
            self.downs.append(Down(ch, ch * 2))
            ch *= 2
            encoder_channels.append(ch)

        self.bottleneck = ConvBlock(ch, ch * 2)
        ch *= 2

        self.ups = nn.ModuleList()
        for skip_ch in reversed(encoder_channels[:-1]):
            self.ups.append(Up(ch, skip_ch, skip_ch))
            ch = skip_ch

        self.outc = nn.Conv2d(ch, out_channels, 1)

    def forward(self, x):
        x1 = self.inc(x)
        skip = [x1]

        for down in self.downs:
            skip.append(down(skip[-1]))

        x = self.bottleneck(skip[-1])
        skip = skip[:-1]

        for up, sk in zip(self.ups, reversed(skip)):
            x = up(x, sk)

        return self.outc(x)
