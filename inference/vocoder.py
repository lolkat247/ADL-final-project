import torch
import torchaudio

class HiFiGAN:
    def __init__(self, checkpoint_path, device='cpu'):
        self.device = device
        self.model = torch.load(checkpoint_path, map_location=device)['generator']
        self.model.eval().to(device)

    def __call__(self, mel):
        with torch.no_grad():
            audio = self.model(mel)
        return audio.squeeze()