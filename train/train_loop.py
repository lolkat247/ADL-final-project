import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import zipfile
import urllib.request
from tqdm import tqdm

from common.model import VoiceAutoencoder
from common.speaker_embed import ECAPASpeakerEmbedder
from train.voice_dataset import VoiceDataset

def download_vctk_subset(destination="data/vctk"):
    url = "https://datashare.ed.ac.uk/bitstream/handle/10283/2651/VCTK-Corpus.zip"
    zip_path = "data/vctk.zip"
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(destination):
        if not os.path.exists(zip_path):
            print("Downloading VCTK Corpus (approx 10GB)...")
            with urllib.request.urlopen(url) as response, open(zip_path, 'wb') as out_file:
                total = int(response.headers.get("Content-Length", 0))
                with tqdm(total=total, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        pbar.update(len(chunk))
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("data")
        print("Done extracting VCTK.")
    else:
        print("VCTK dataset already exists.")

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = VoiceAutoencoder().to(device=device)
    embedder = ECAPASpeakerEmbedder(device=device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Download dataset
    download_vctk_subset()

    # Load dataset
    root_dir = "data/vctk/wav48_silence_trimmed"
    paths, ids = [], []

    for speaker_folder in os.listdir(root_dir):
        speaker_path = os.path.join(root_dir, speaker_folder)
        if not os.path.isdir(speaker_path): continue

        speaker_id = speaker_folder  # e.g., 'p225'
        wav_files = [f for f in os.listdir(speaker_path) if f.endswith('.wav')]
        for wav_file in wav_files:
            paths.append(os.path.join(speaker_path, wav_file))
            ids.append(speaker_id)

    dataset = VoiceDataset(paths, ids)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Training loop
    for epoch in range(20):
        model.train()
        for mel, speaker_id in loader:
            mel = mel.to(device)

            # Use speaker ID to simulate reference audio embedding (you may want actual audio files instead)
            speaker_embedding = embedder.extract_embedding(speaker_id)
            speaker_embedding = speaker_embedding.unsqueeze(0).expand(mel.size(0), -1).to(device)

            output = model(mel, speaker_embedding)
            loss = criterion(output, mel)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

if __name__ == "__main__":
    main()