import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import UNet


class DCConv(nn.Module):
    def __init__(self, weight_init=0.1):
        super().__init__()
        self.lambda_ = nn.Parameter(torch.tensor(weight_init))

    def forward(self, x_recon, kspace_us, mask):
        x_recon = x_recon.squeeze(1)
        batch = x_recon.shape[0]
        out = []
        for b in range(batch):
            img = x_recon[b]
            ksp = torch.fft.fftshift(torch.fft.fft2(img))
            ksp_dc = mask * kspace_us + (1 - mask) * ksp
            dc_img = torch.fft.ifft2(torch.fft.ifftshift(ksp_dc)).real
            out.append((1 - self.lambda_.abs()) * img + self.lambda_.abs() * dc_img)
        return torch.stack(out).unsqueeze(1)


class UnrolledReconNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base_channels=32, depth=4,
                 num_cascades=5, dc_weight=0.1):
        super().__init__()
        self.num_cascades = num_cascades

        self.denoisers = nn.ModuleList([
            UNet(in_channels if i == 0 else 1, out_channels, base_channels, depth)
            for i in range(num_cascades)
        ])

        self.dc_layers = nn.ModuleList([
            DCConv(dc_weight) for _ in range(num_cascades)
        ])

    def forward(self, aliased, t1_full=None, kspace_us=None, mask=None):
        if t1_full is not None:
            x = torch.cat([aliased, t1_full], dim=1)
        else:
            x = aliased

        for i in range(self.num_cascades):
            x = x[:, :1] if x.shape[1] > 1 and i > 0 else x
            x = self.denoisers[i](x)
            if i < self.num_cascades - 1:
                x = self.dc_layers[i](x, kspace_us, mask)

        return x
