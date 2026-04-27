import torch
import torch.nn as nn
from .unet import UNet


class DCConv(nn.Module):
    def __init__(self, weight_init=0.1):
        super().__init__()
        self.lambda_ = nn.Parameter(torch.tensor(weight_init))

    def forward(self, x_recon, kspace_us, mask):
        device_type = x_recon.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            x_recon = x_recon.float()
            x_recon_2d = x_recon.squeeze(1)
            predicted_kspace = torch.fft.fftshift(torch.fft.fft2(x_recon_2d), dim=(-2, -1))
            measured_kspace = torch.complex(kspace_us[:, 0].float(), kspace_us[:, 1].float())
            sampling_mask = mask.squeeze(1).float()

            kspace_dc = sampling_mask * measured_kspace + (1 - sampling_mask) * predicted_kspace
            dc_img = torch.fft.ifft2(torch.fft.ifftshift(kspace_dc, dim=(-2, -1))).real.unsqueeze(1)

            blend = self.lambda_.abs().float()
            return (1 - blend) * x_recon + blend * dc_img


class UnrolledReconNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base_channels=32, depth=4,
                 num_cascades=5, dc_weight=0.1):
        super().__init__()
        self.num_cascades = num_cascades
        self.use_multimodal = in_channels > 1

        self.denoisers = nn.ModuleList([
            UNet(in_channels if self.use_multimodal else 1, out_channels, base_channels, depth)
            for i in range(num_cascades)
        ])

        self.dc_layers = nn.ModuleList([
            DCConv(dc_weight) for _ in range(num_cascades)
        ])

    def forward(self, aliased, t1_full=None, kspace_us=None, mask=None):
        x = aliased

        for i in range(self.num_cascades):
            if self.use_multimodal:
                if t1_full is None:
                    raise ValueError('t1_full is required when in_channels > 1')
                denoiser_input = torch.cat([x, t1_full], dim=1)
            else:
                denoiser_input = x

            x = self.denoisers[i](denoiser_input)
            if kspace_us is not None and mask is not None:
                x = self.dc_layers[i](x, kspace_us, mask)

        return x
