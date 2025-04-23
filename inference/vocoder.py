import torch
from models.hifigan import Generator

class HiFiGAN:
    def __init__(self, checkpoint_path, device='cpu'):
        self.device = device
        self.model = Generator().to(device)
        self.load_weights(checkpoint_path)

    def load_weights(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['generator'])
        self.model.eval()

    def __call__(self, mel):
        mel = mel.to(self.device)
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)  # add batch dimension
        with torch.no_grad():
            audio = self.model(mel)  # [B, T]
        return audio.squeeze()
