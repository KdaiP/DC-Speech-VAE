import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn as nn
import torchaudio

from models.model import DCAEWrapper, remove_all_weight_norms
from models.vocoder_wrapper import VocoderWrapper

class DCAEInferenceWrapper(nn.Module):
    def __init__(self, ae_model_path, ae_config_path, vocoder_model_path):
        super().__init__()
        # load dcae model
        self.ae_model = DCAEWrapper(ae_config_path)
        self.ae_model.load_state_dict(torch.load(ae_model_path, map_location='cpu', weights_only=True))
        self.ae_model.eval()
        remove_all_weight_norms(self.ae_model)

        # load vocoder model and mel transform
        self.vocoder_model = VocoderWrapper(vocoder_model_path)
        self.transform, self.sample_rate = self.vocoder_model.transform, self.vocoder_model.sample_rate

        self.padding_factor = 256 * 16

        # get the output channel from config
        # we reshape 2d latent to 1d latent: [b. latent_channel, mel_comp_channel, t] -> [b, latent_channel * mel_comp_channel, t]
        self.latent_channel = self.ae_model.model.config.latent_channels
        self.mel_comp_channel = self.vocoder_model.model.h.num_mels // (2 ** (len(self.ae_model.model.config.encoder_block_out_channels) - 1))

    @torch.no_grad()
    def forward(self, audio, sample_rate):
        latent = self.encode(audio, sample_rate)
        reconstructed_audio = self.decode(latent)
        return reconstructed_audio

    def encode(self, audio, sample_rate, return_1d=True):
        """
        Encode audio to latent representation.
        Input audio shape: [b, t]
        """
        device = next(self.ae_model.parameters()).device

        if sample_rate != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sample_rate, self.sample_rate)
        if audio.size(-1) % self.padding_factor != 0:
            pad_size = self.padding_factor - (audio.size(-1) % self.padding_factor)
            audio = torch.nn.functional.pad(audio, (0, pad_size), mode='constant', value=0)

        feature = self.transform(audio).to(device) # [b, c, t]
        latent = self.ae_model.encode(feature) # [b, c1, c2, t]

        if return_1d:
            b, c1, c2, t = latent.size()
            latent = latent.reshape(b, c1 * c2, t)
        return latent

    def decode(self, latent):
        """
        Decode latent representation to audio.
        Input latent shape: [b, c, t] or [b, c1, c2, t]
        """
        if latent.ndim == 2:
            latent = latent.unsqueeze(0)
        if latent.ndim == 3:
            latent = latent.reshape(latent.size(0), self.latent_channel, self.mel_comp_channel, latent.size(2))

        reconstructed_feature = self.ae_model.decode(latent)
        reconstructed_audio = self.vocoder_model.forward_model(reconstructed_feature)
        return reconstructed_audio # [b, t]

if __name__ == "__main__":
    model_config = {
    'ae_model_path': 'checkpoints/generator_700000.pt',
    'ae_config_path': 'models/default_config.json',
    'vocoder_model_path': 'models/bigvgan_v2_22khz_80band_256x',
    }

    audio_input_path = 'path/to/input/audio/file.wav'
    audio_output_path = 'path/to/output/audio/file.wav'

    device = 'cuda'  # or 'cpu'
    model = DCAEInferenceWrapper(**model_config).to(device)

    audio, sample_rate = torchaudio.load(audio_input_path)
    audio = audio.to(device)

    with torch.inference_mode():
        reconstructed_audio = model(audio, sample_rate).cpu()

    torchaudio.save(audio_output_path, reconstructed_audio, model.sample_rate)