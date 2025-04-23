import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import zipfile
import urllib.request
from tqdm import tqdm
import torch.nn.functional as F

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
    extract_path = os.path.join(destination, "VCTK-Corpus")
    if not os.path.exists(extract_path):
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(destination)
        print("Done extracting VCTK.")
    else:
        print("VCTK corpus already extracted.")

def collate_fn(batch):
    mels, speaker_ids = zip(*batch)
    padded_mels = pad_sequence(mels, batch_first=True)  # [B, T, 80]
    return padded_mels, speaker_ids

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = VoiceAutoencoder().to(device=device)
    embedder = ECAPASpeakerEmbedder(device=device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Download dataset
    download_vctk_subset()

    # Load dataset
    root_dir = "data/vctk/VCTK-Corpus/wav48"
    paths, ids = [], []

    # Limit to 10 speakers
    for speaker_folder in os.listdir(root_dir)[:10]:
        speaker_path = os.path.join(root_dir, speaker_folder)
        if not os.path.isdir(speaker_path): continue

        speaker_id = speaker_folder  # e.g., 'p225'
        # Limit to 3 WAV files per speaker
        wav_files = [f for f in os.listdir(speaker_path) if f.endswith('.wav')][:3]
        for wav_file in wav_files:
            paths.append(os.path.join(speaker_path, wav_file))
            ids.append(speaker_id)
        wav_files = [f for f in wav_files if f.endswith('.wav')]

    # Print summary of loaded data
    print(f"Loaded {len(paths)} samples from {len(set(ids))} speakers.")

    # Build reference map: speaker_id -> first wav path
    speaker_to_ref = {}
    for path, sid in zip(paths, ids):
        if sid not in speaker_to_ref:
            speaker_to_ref[sid] = path

    # Precompute and cache speaker embeddings
    embedding_cache_path = "cache/speaker_embeddings.pt"
    os.makedirs("cache", exist_ok=True)

    if os.path.exists(embedding_cache_path):
        print("Loading cached speaker embeddings...")
        speaker_embeddings_map = torch.load(embedding_cache_path)
    else:
        print("Computing speaker embeddings...")
        speaker_embeddings_map = {
            sid: embedder.extract_embedding(ref_path).squeeze().to(device)
            for sid, ref_path in speaker_to_ref.items()
        }
        torch.save(speaker_embeddings_map, embedding_cache_path)

    dataset = VoiceDataset(paths, ids)
    # Use batch size 2
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    # Training loop
    print("Starting training loop...")
    # Train for 3 epochs
    for epoch in range(3):
        model.train()
        total_loss = 0.0
        total_cosine = 0.0
        num_batches = 0
        for mel, speaker_ids in tqdm(loader, desc=f"Epoch {epoch}"):
            mel = mel.to(device)
            speaker_embeddings = torch.stack([
                speaker_embeddings_map[sid]
                for sid in speaker_ids
            ]).to(device)

            output = model(mel, speaker_embeddings)
            loss = criterion(output, mel)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Compute speaker similarity
            with torch.no_grad():
                out_wav = output.transpose(1, 2)  # [B, 80, T]
                out_embed = torch.stack([
                    embedder.extract_embedding_from_mel(mel.cpu()).squeeze()
                    for mel in out_wav
                ]).to(device)

                cosine_scores = F.cosine_similarity(out_embed, speaker_embeddings, dim=-1)
                total_cosine += cosine_scores.mean().item()

            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_cosine = total_cosine / num_batches
        print(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}, Avg Cosine = {avg_cosine:.4f}", flush=True)

if __name__ == "__main__":
    main()