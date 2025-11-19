import os
import json

import torch
import torch.nn as nn

from diffusers import AutoencoderDC

def load_dcae_model(model_path, **kwargs) -> AutoencoderDC:
    """
    load DCAE model from a directory or a config file
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model path does not exist: {model_path}')
    
    if os.path.isdir(model_path):
        model = AutoencoderDC.from_pretrained(model_path, **kwargs)

    elif model_path.lower().endswith('.json'):
        with open(model_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        model = AutoencoderDC.from_config(config)

    else:
        raise ValueError(f'Invalid model_path: {model_path}')

    # add weight norm to all conv2d layers to stabilize training
    model = add_weight_norm_to_all_2d_convs(model)
    
    return model

def sample_sigma_vae(x, noise_strength=0, dist_type='fix'):
    """
    add noise to encoder out during training
    modified from:
    https://github.com/microsoft/unilm/blob/master/LatentLM/tokenizer_models/modeling_sigma_vae.py
    https://github.com/stepfun-ai/NextStep-1/blob/main/nextstep/models/modeling_flux_vae.py
    """
    if noise_strength == 0:
        return x

    noise = torch.randn_like(x)
    if dist_type == 'fix':
        x = x + noise_strength * noise
    elif dist_type == 'gaussian':
        std = torch.rand([x.size(0), 1, 1, 1], device=x.device, dtype=x.dtype) * noise_strength
        x = x + std * noise
    else:
        raise ValueError(f'Unknown dist_type: {dist_type}')
    return x

def compute_dispersive_loss(x):
    """
    dispersive loss:
    https://arxiv.org/pdf/2506.09027
    modified from:
    https://github.com/zhangq327/U-MAE/blob/main/loss_func.py
    https://github.com/raywang4/DispLoss
    """
    x = x.reshape(x.size(0), -1)
    b, c_t = x.shape
    x = torch.nn.functional.normalize(x, p=2, dim=-1)

    sim = torch.matmul(x, x.transpose(0, 1))
    mask = 1 - torch.eye(b, device=x.device)

    loss = (sim.pow(2) * mask).sum() / mask.sum()
    return loss

def add_weight_norm_to_all_2d_convs(module: nn.Module) -> nn.Module:
    """
    add weight norm to all conv2d and convtranspose2d layers
    """
    from torch.nn.utils.parametrize import is_parametrized
    from torch.nn.utils.parametrizations import weight_norm
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if is_parametrized(m, 'weight'):
                continue
            weight_norm(m)
    return module

def remove_all_weight_norms(module: nn.Module, verbose=False):
    """
    remove weight norm from all conv2d and convtranspose2d layers during inference
    """
    from torch.nn.utils.parametrize import remove_parametrizations, is_parametrized
    for name, submodule in module.named_children():
        remove_all_weight_norms(submodule)
        
        if is_parametrized(submodule, 'weight'):
            remove_parametrizations(submodule, 'weight')
            if verbose:
                print(f"Removed weight norm from {name}")
    return module

class DCAEWrapper(nn.Module):
    def __init__(self, config_path):
        super().__init__()
        self.model = load_dcae_model(config_path)
        self.norm = nn.LayerNorm(self.model.config.latent_channels, elementwise_affine=False)

    def print_num_params(self):
        print('Encoder:', sum(p.numel() for p in self.model.encoder.parameters()) / 1e6, "M parameters")
        print('Decoder:', sum(p.numel() for p in self.model.decoder.parameters()) / 1e6, "M parameters")

    def encode(self, x, noise_strength=0):
        if x.ndim == 3:
            x = x.unsqueeze(1)

        latent = self.model.encode(x).latent
        latent = self.norm(latent.transpose(1, -1)).transpose(1, -1)

        if self.training:
            dist_type = 'gaussian'
            latent_clean = latent.clone()
        else:
            dist_type = 'fix'

        latent = sample_sigma_vae(latent, noise_strength, dist_type)

        if self.training:
            return latent, latent_clean

        return latent

    def decode(self, x):
        sample = self.model.decode(x).sample
        return sample.squeeze(1)

    def forward(self, input):
        """forward is used only for training"""
        latent, latent_clean = self.encode(input, noise_strength=0.5)
        sample = self.decode(latent)
        return sample, latent_clean

if __name__ == "__main__":
    device = torch.device("cuda")

    model = DCAEWrapper("models/default_config.json").to(device)
    model.print_num_params()

    x = torch.randn(2, 1, 80, 160, device=device)

    latent, latent_clean = model.encode(x)
    print(f"latent shape: {latent.shape}")

    y = model.decode(latent)
    model.model.save_pretrained("checkpoints/test")

    loss, loss_dict = model(x)
    print(f"loss: {loss}, loss_dict: {loss_dict}")