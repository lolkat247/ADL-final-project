import torch
from models.hifigan import Generator  # assuming HiFi-GAN code is in models/
from utils import load_checkpoint     # utility to load weights

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
        with torch.no_grad():
            audio = self.model(mel)  # [B, T]
        return audio
