import torch
from common.mel_spectrogram import MelSpectrogram
from common.model import VoiceAutoencoder
from inference.vocoder import HiFiGAN
from common.speaker_embed import DummySpeakerEmbedder
import torchaudio

class InferencePipeline:
    def __init__(self, ae_checkpoint, vocoder_checkpoint, device='cpu'):
        self.device = device
        self.mel_transform = MelSpectrogram()
        self.model = VoiceAutoencoder().to(device)
        self.model.load_state_dict(torch.load(ae_checkpoint))
        self.model.eval()

        self.vocoder = HiFiGAN(vocoder_checkpoint, device)
        self.embedder = DummySpeakerEmbedder()

    def convert(self, audio, source_sr, target_speaker_id):
        if isinstance(audio, str):
            audio, source_sr = torchaudio.load(audio)

        audio = torchaudio.functional.resample(audio, source_sr, 22050).to(self.device)
        mel = self.mel_transform(audio).transpose(1, 2)  # (B, T, F)
        embed = self.embedder(target_speaker_id).to(self.device)

        with torch.no_grad():
            output_mel = self.model(mel, embed)
            audio_out = self.vocoder(output_mel.transpose(1, 2))

        return audio_out