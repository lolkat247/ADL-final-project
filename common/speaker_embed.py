import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier

class ECAPASpeakerEmbedder:
    def __init__(self, device='cpu'):
        self.device = device
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained/ecapa",
            run_opts={"device": device}
        )

    def extract_embedding(self, wav_path):
        """Load WAV file and return speaker embedding"""
        signal, fs = torchaudio.load(wav_path)
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)  # mono
        signal = signal.to(self.device)
        with torch.no_grad():
            embedding = self.classifier.encode_batch(signal).squeeze(0)
        return embedding  # Shape: [embedding_dim]