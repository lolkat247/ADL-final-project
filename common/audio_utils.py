import librosa
import soundfile as sf

def load_audio(path, sr=22050):
    audio, _ = librosa.load(path, sr=sr)
    return audio

def save_audio(path, audio, sr=22050):
    sf.write(path, audio, sr)