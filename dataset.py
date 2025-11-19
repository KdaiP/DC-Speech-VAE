import os
import random

import torch
import torchaudio

from tqdm import tqdm

from models.vocoder_wrapper import VocoderWrapper

def chunk_and_resample_audio(audio, sample_rate, target_sample_rate, target_samples):
    if audio.ndim != 2:
        raise ValueError(f"Expected audio shape (channels, samples), got {audio.shape}")
    
    if sample_rate != target_sample_rate:
        audio = torchaudio.functional.resample(audio, sample_rate, target_sample_rate)

    audio_total_samples = audio.size(-1)
    start_ratio = 0.0
    
    if audio_total_samples > target_samples:
        max_start = audio_total_samples - target_samples
        start = random.randint(0, max_start)
        audio = audio[:, start:start + target_samples]
        start_ratio = start / audio_total_samples

    elif audio_total_samples < target_samples:
        pad_size = target_samples - audio_total_samples
        audio = torch.nn.functional.pad(audio, (0, pad_size))
        start_ratio = 0.0
    
    return audio, start_ratio

class AEDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, vocoder_config_path, segment_samples, verbose=True):
        self.verbose = verbose
        self.segment_samples = segment_samples

        self.transform, self.sample_rate = VocoderWrapper.get_feature_extractor(vocoder_config_path)
        self.samples = self.load_filelist(data_path)

        if verbose:
            print(f"Initialized AE Dataset with {len(self.samples)} samples.")
            print(f"Sample rate: {self.sample_rate}, Segment samples: {self.segment_samples}")
        
    def load_filelist(self, data_path):
        samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Loading filelist", unit="lines"):
                if len(line.strip()) > 0:
                    samples.append(line.strip())
        return samples

    def get_feature_from_path(self, audio_path):
        audio, sample_rate = torchaudio.load(audio_path)
        audio = chunk_and_resample_audio(audio, sample_rate, self.sample_rate, self.segment_samples)[0]
        feature = self.transform(audio).squeeze(0) # [b, t]

        if self.verbose:
            print(f"Loaded audio from {audio_path}")
            print(f"audio shape: {audio.shape}, feature shape: {feature.shape}")
        return feature

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path = self.samples[idx]
        feature = self.get_feature_from_path(audio_path)
        return [feature]

class AEDatasetCollate:
    def __init__(self, verbose=True) -> None:
        self.verbose = verbose

    def __call__(self, batch):
        features,  = map(list, zip(*(row for row in batch)))
        features = torch.stack(features, dim=0) # [b, c, t]

        if self.verbose:
            print(f"Collated batch size: {len(batch)}, feature shape: {features.shape}")
        return [features]


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    data_path = 'filelists/filelist.txt'
    vocoder_config_path = 'models/bigvgan_v2_22khz_80band_256x/config.json'

    dataset = AEDataset(data_path, vocoder_config_path, 86016, verbose=True)
    data = next(iter(dataset))
    print(data)
    loader = DataLoader(dataset, 8, collate_fn=AEDatasetCollate(True))
    for idx, datas in enumerate(loader):
        if idx > 1:
            break
