# inference/inference_pipeline.py

import torch
import torchaudio
from common.model import VoiceAutoencoder
from common.mel_spectrogram import MelSpectrogram
from common.speaker_embed import ECAPASpeakerEmbedder
from inference.vocoder import HiFiGAN


class InferencePipeline:
    def __init__(self, ae_ckpt_path, hifi_ckpt_path, device='cuda'):
        self.device = device

        # Load autoencoder
        self.model = VoiceAutoencoder().to(device)
        self.model.load_state_dict(torch.load(ae_ckpt_path, map_location=device))
        self.model.eval()

        # Load vocoder
        self.vocoder = HiFiGAN(hifi_ckpt_path, device)

        # Feature extractor and speaker embedder
        self.mel_extractor = MelSpectrogram().to(device)
        self.embedder = ECAPASpeakerEmbedder(device=device)

    def convert(self, input_wav, ref_wav):
        """
        input_wav: path to source speaker audio file
        ref_wav: path to target speaker reference audio
        """
        with torch.no_grad():
            # Load and process input audio
            wav, sr = torchaudio.load(input_wav)
            wav = torchaudio.functional.resample(wav, sr, 22050)
            wav = wav.to(self.device)

            # Extract Mel
            mel = self.mel_extractor(wav).squeeze(0).transpose(0, 1).unsqueeze(0)  # [1, T, 80]

            # Get speaker embedding
            speaker_embedding = self.embedder.extract_embedding(ref_wav).unsqueeze(0)  # [1, 192]

            # Convert Mel
            mel_out = self.model(mel, speaker_embedding)  # [1, T, 80]
            mel_out = mel_out.transpose(1, 2)  # [1, 80, T]

            # Generate audio
            audio_out = self.vocoder(mel_out)  # [1, T]

        return audio_out.squeeze(0).cpu()
