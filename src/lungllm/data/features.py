# REPO PATH: src/lungllm/data/features.py
"""Audio loading (soundfile, no torchcodec) + log-mel features."""
from __future__ import annotations


def load_audio(path, target_sr=16000):
    import soundfile as sf
    import torch
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.mean(axis=1)).contiguous()
    if sr != target_sr:
        import torchaudio
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def crop_or_pad(waveform, num_samples):
    import torch
    T = waveform.shape[-1]
    if T == num_samples: return waveform
    if T > num_samples:
        s = (T - num_samples) // 2
        return waveform[s:s + num_samples]
    return torch.nn.functional.pad(waveform, (0, num_samples - T))


class LogMelExtractor:
    def __init__(self, sample_rate=16000, n_mels=128, n_fft=1024, hop_length=160,
                 win_length=400, target_seconds=8.0, top_db=80.0):
        self.sample_rate = sample_rate; self.n_mels = n_mels; self.n_fft = n_fft
        self.hop_length = hop_length; self.win_length = win_length
        self.target_seconds = target_seconds; self.top_db = top_db; self._mel = None

    def _build(self):
        import torchaudio
        self._mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate, n_fft=self.n_fft, win_length=self.win_length,
            hop_length=self.hop_length, n_mels=self.n_mels, power=2.0)
        self._to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=self.top_db)

    def num_samples(self): return int(self.target_seconds * self.sample_rate)

    def __call__(self, waveform):
        if self._mel is None: self._build()
        waveform = crop_or_pad(waveform, self.num_samples())
        logmel = self._to_db(self._mel(waveform))
        return (logmel - logmel.mean()) / (logmel.std() + 1e-5)
