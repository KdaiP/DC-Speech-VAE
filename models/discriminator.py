from typing import List, Tuple

import torch
from torch import nn
from torch import Tensor
from torch.nn.utils.parametrizations import weight_norm

# losses are copied from:
# https://github.com/jik876/hifi-gan/blob/master/models.py

def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses

def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses

class AEDiscriminator(nn.Module):
    def __init__(
        self,
        n_mel_bins: int = 80,
    ):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorR(n_mel_bins=n_mel_bins), DiscriminatorPatchGAN()]
        )

    def forward(self, y: Tensor, y_hat: Tensor) -> Tuple[List[Tensor], List[Tensor], List[List[Tensor]], List[List[Tensor]]]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for d in self.discriminators:
            y_d_r, fmap_r = d(x=y)
            y_d_g, fmap_g = d(x=y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class DiscriminatorR(nn.Module):
    def __init__(
        self,
        n_mel_bins: int = 80,
        bands: Tuple[Tuple[float, float], ...] = ((0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)),
    ):
        """
        modified from:
        https://github.com/descriptinc/descript-audio-codec
        bands split suggested by:
        https://github.com/ylzz1997
        """
        super().__init__()
        bands = [(int(b[0] * n_mel_bins), int(b[1] * n_mel_bins)) for b in bands]
        self.bands = bands
        convs = lambda: nn.ModuleList(
            [
                weight_norm(nn.Conv2d(1, 32, (3, 3), (1, 1), padding=(1, 1))),
                weight_norm(nn.Conv2d(32, 32, (3, 3), (1, 1), padding=(1, 1))),
                weight_norm(nn.Conv2d(32, 32, (3, 3), (1, 1), padding=(1, 1))),
                weight_norm(nn.Conv2d(32, 32, (3, 3), (1, 1), padding=(1, 1))),
                weight_norm(nn.Conv2d(32, 32, (3, 3), (1, 1), padding=(1, 1))),
            ]
        )
        self.band_convs = nn.ModuleList([convs() for _ in range(len(self.bands))])

        self.conv_post = weight_norm(nn.Conv2d(32, 1, (3, 3), (1, 1), padding=(1, 1)))

    def spectrogram(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1).transpose(2, 3)  # b f t -> b 1 t f
        else:
            raise ValueError(f"Expected input shape (b, c, t), got {x.shape}")
        # Split into bands
        x_bands = [x[..., b[0] : b[1]] for b in self.bands]
        return x_bands

    def forward(self, x: Tensor):
        x_bands = self.spectrogram(x)
        fmap = []
        x = []
        for band, stack in zip(x_bands, self.band_convs):
            for i, layer in enumerate(stack):
                band = layer(band)
                band = torch.nn.functional.leaky_relu(band, 0.1, inplace=True)
                if i > 0:
                    fmap.append(band)
            x.append(band)
        x = torch.cat(x, dim=-1)
        x = self.conv_post(x)
        fmap.append(x)

        return x, fmap

class DiscriminatorPatchGAN(nn.Module):
    def __init__(self):
        """
        A PatchGAN discriminator as in Pix2Pix:
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py

        This version is modified from:
        https://github.com/jik876/hifi-gan/blob/master/models.py
        https://github.com/CompVis/taming-transformers/blob/master/taming/modules/discriminator/model.py
        """
        super().__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(1, 32, 4, 2, padding=1)),
                weight_norm(nn.Conv2d(32, 64, 4, 2, padding=1)),
                weight_norm(nn.Conv2d(64, 128, 4, 2, padding=1)),
                weight_norm(nn.Conv2d(128, 256, 4, 2, padding=1)),
                weight_norm(nn.Conv2d(256, 512, 4, 2, padding=1)),
            ]
        )
        self.conv_post = weight_norm(nn.Conv2d(512, 1, 4, 1, padding=1))

    def forward(self, x: Tensor):
        if x.ndim == 3:
            x = x.unsqueeze(1).transpose(2, 3)  # b f t -> b 1 t f
        else:
            raise ValueError(f"Expected input shape (b, c, t), got {x.shape}")
        fmap = []
        for i, layer in enumerate(self.convs):
            x = layer(x)
            x = torch.nn.functional.leaky_relu(x, 0.1, inplace=True)
            if i > 0:
                fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)

        return x, fmap
