import torchaudio
from torch.utils.data import Dataset
from common.mel_spectrogram import MelSpectrogram

class VoiceDataset(Dataset):
    def __init__(self, file_paths, speaker_ids):
        self.paths = file_paths
        self.ids = speaker_ids
        self.mel_transform = MelSpectrogram()

    def __getitem__(self, index):
        audio, sr = torchaudio.load(self.paths[index])
        audio = torchaudio.functional.resample(audio, sr, 22050)
        mel = self.mel_transform(audio).squeeze().transpose(0, 1)
        return mel, self.ids[index]

    def __len__(self):
        return len(self.paths)