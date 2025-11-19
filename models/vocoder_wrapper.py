import torch
import torch.nn as nn
import torchaudio

class VocoderWrapper(nn.Module):
    def __init__(self, model_path, **kwargs):
        super().__init__()
        from .bigvgan_v2_22khz_80band_256x import bigvgan
        DEFAULT_MODEL_PATH = 'nvidia/bigvgan_v2_22khz_80band_256x'
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH

        self.model = bigvgan.BigVGAN.from_pretrained(model_path, **kwargs)
        self.model.remove_weight_norm()
        self.model.eval()

        self.transform, self.sample_rate = self.get_feature_extractor(self.model.h)

    @staticmethod
    def get_feature_extractor(config):
        from functools import partial
        from .bigvgan_v2_22khz_80band_256x.meldataset import get_mel_spectrogram
        from .bigvgan_v2_22khz_80band_256x.bigvgan import load_hparams_from_json
        from .bigvgan_v2_22khz_80band_256x.env import AttrDict
        
        if not isinstance(config, AttrDict):
            config = load_hparams_from_json(config)

        transform = partial(get_mel_spectrogram, h=config)
        sample_rate = config.sampling_rate

        return transform, sample_rate

    def forward_model(self, x):
        return self.model(x).squeeze(1) # [b, t]

    @ torch.no_grad()
    def forward(self, audio, sample_rate):
        device = next(self.model.parameters()).device

        if sample_rate != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sample_rate, self.sample_rate)

        feature = self.transform(audio).to(device) # [b, c, t]
        return self.forward_model(feature)