import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=256, num_layers=2):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        _, (hidden, _) = self.rnn(x)
        return hidden[-1]

class Decoder(nn.Module):
    def __init__(self, hidden_dim=256, speaker_dim=128, output_dim=80, num_layers=2):
        super().__init__()
        self.input_fc = nn.Linear(hidden_dim + speaker_dim, hidden_dim)
        self.rnn = nn.LSTM(hidden_dim, output_dim, num_layers, batch_first=True)

    def forward(self, content, speaker_embed, seq_len):
        x = torch.cat([content, speaker_embed], dim=1).unsqueeze(1).repeat(1, seq_len, 1)
        x = self.input_fc(x)
        output, _ = self.rnn(x)
        return output

class VoiceAutoencoder(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=256, speaker_dim=128):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, speaker_dim, input_dim)

    def forward(self, x, speaker_embed):
        content = self.encoder(x)
        output = self.decoder(content, speaker_embed, x.size(1))
        return output