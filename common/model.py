# common/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContentEncoder(nn.Module):
    def __init__(self, mel_dim=80, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(mel_dim, hidden_dim, kernel_size=5, padding=4, dilation=2),  # causal-style
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=4, dilation=2),
            nn.ReLU(),
        )

    def forward(self, mel):
        mel = mel.transpose(1, 2)  # [B, mel_dim, T]
        content = self.net(mel)  # [B, hidden_dim, T]
        return content.transpose(1, 2)  # [B, T, hidden_dim]


class ContentAttention(nn.Module):
    def __init__(self, content_dim, speaker_dim, attn_dim):
        super().__init__()
        self.content_proj = nn.Linear(content_dim, attn_dim)
        self.speaker_proj = nn.Linear(speaker_dim, attn_dim)
        self.energy_proj = nn.Linear(attn_dim, 1)

    def forward(self, content, speaker_embedding):
        # content: [B, T, C], speaker_embedding: [B, S]
        B, T, _ = content.size()
        speaker_exp = self.speaker_proj(speaker_embedding).unsqueeze(1).expand(-1, T, -1)
        content_proj = self.content_proj(content)
        energy = torch.tanh(content_proj + speaker_exp)  # [B, T, attn_dim]
        attn_weights = torch.softmax(self.energy_proj(energy), dim=1)  # [B, T, 1]
        attended = content * attn_weights  # weighted content
        return attended


class Decoder(nn.Module):
    def __init__(self, input_dim, mel_dim=80):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(256, mel_dim, kernel_size=5, padding=2),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        mel_out = self.net(x)
        return mel_out.transpose(1, 2)


class VoiceAutoencoder(nn.Module):
    def __init__(self, mel_dim=80, content_dim=256, speaker_dim=192, attn_dim=128):
        super().__init__()
        self.encoder = ContentEncoder(mel_dim, content_dim)
        self.attn = ContentAttention(content_dim, speaker_dim, attn_dim)
        self.decoder = Decoder(content_dim, mel_dim)

    def forward(self, mel, speaker_embedding):
        content = self.encoder(mel)  # [B, T, C]
        attended = self.attn(content, speaker_embedding)  # [B, T, C]
        mel_out = self.decoder(attended)  # [B, T, mel_dim]
        return mel_out
