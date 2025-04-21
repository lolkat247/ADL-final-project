import torch
import torchaudio
from torch.utils.data import Dataset
from common.mel_spectrogram import MelSpectrogram

class VoiceDataset(Dataset):
    def __init__(self, file_paths, speaker_ids, chunk_size=96):
        self.paths = file_paths
        self.ids = speaker_ids
        self.chunk_size = chunk_size
        self.mel_transform = MelSpectrogram()

    def __getitem__(self, index):
        audio, sr = torchaudio.load(self.paths[index])
        audio = torchaudio.functional.resample(audio, sr, 22050)
        mel = self.mel_transform(audio).squeeze().transpose(0, 1)  # [T, 80]

        T = mel.size(0)
        if T < self.chunk_size:
            pad = torch.zeros(self.chunk_size - T, mel.size(1))
            mel = torch.cat([mel, pad], dim=0)
        elif T > self.chunk_size:
            start = torch.randint(0, T - self.chunk_size + 1, (1,)).item()
            mel = mel[start:start + self.chunk_size]

        return mel, self.ids[index]

    def __len__(self):
        return len(self.paths)