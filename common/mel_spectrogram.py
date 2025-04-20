import torchaudio.transforms as T

class MelSpectrogram:
    def __init__(self, sample_rate=22050, n_fft=1024, hop_length=256, n_mels=80):
        self.transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        self.amplitude_to_db = T.AmplitudeToDB()

    def __call__(self, waveform):
        mel = self.transform(waveform)
        return self.amplitude_to_db(mel)