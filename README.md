# Voice Conversion with Autoencoder + HiFi-GAN

This project aims to perform voice conversion using a content encoder + speaker embedding architecture and HiFi-GAN for waveform synthesis. The goal is to take the **content of one speaker's voice** and **convert it into the sound of another speaker**.

## Overview

The system consists of three main components:

1. **Autoencoder (AE)**  
   - Encoder: Extracts content representation from Mel-spectrograms  
   - Decoder: Reconstructs Mel-spectrograms using content + target speaker embedding

2. **Speaker Embeddings**  
   - We use ECAPA-TDNN from SpeechBrain to extract speaker embeddings from a short reference clip of the target voice.
   - This allows the model to generate Mel-spectrograms that match the voice characteristics of any speaker given just a few seconds of their speech.
   - See `common/speaker_embed.py` for the implementation.

3. **HiFi-GAN Vocoder**  
   - Converts Mel-spectrograms back into high-quality audio

## Project Structure
```
voice-conversion/
├── common/                # Shared utilities
│   ├── audio_utils.py     # Load/save audio
│   ├── mel_spectrogram.py # Feature extraction
│   ├── speaker_embed.py   # Speaker embedding module
│   └── model.py           # AE model architecture
│
├── inference/             # Inference and real-time conversion
│   ├── inference_pipeline.py
│   ├── vocoder.py
│   ├── convert.py
│   └── realtime.py
│
├── training/                # Model training
│   └── training_loop.ipynb
│
└── README.md
```
## Quickstart

### Clone Repo & Install Dependencies

```bash
git clone https://github.com/yourusername/voice-conversion.git
cd voice-conversion
pip install -r requirements.txt
```
### Train the Autoencoder

Open and run the training notebook: `training/training_loop.ipynb`

### Convert Audio
```bash
python inference/convert.py \
  --input audio.wav \
  --output converted.wav \
  --ae_ckpt checkpoints/autoencoder.pth \
  --vocoder_ckpt checkpoints/hifigan.pt \
  --speaker_id 0
```

### Real-Time Conversion

We’ll use sounddevice or pyaudio to stream mic → model → speaker.

## 🧪 Architecture Summary
```
             ┌──────────────┐
 Audio Input │              │
 ───────────▶│MelSpectrogram│
             └─────┬────────┘
                   ▼
             ┌──────────────┐
             │   Encoder    │───────┐
             └──────────────┘       │
                                    ▼
                         ┌──────────────────┐
 Speaker ID ────────────▶│ Speaker Embedder │
                         └──────────────────┘
                                    │
                                    ▼
                         ┌──────────────────┐
                         │     Decoder      │
                         └────────┬─────────┘
                                  ▼
                         ┌──────────────────┐
                         │     HiFi-GAN     │
                         └────────┬─────────┘
                                  ▼
                         Synthesized Output
```

## 🙏 Credits
* HiFi-GAN
* LibriTTS Dataset
* VCTK Corpus