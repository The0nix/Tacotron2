from typing import Union, Sequence, List
import pickle

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchnlp.encoders.text


class MelSpectrogram(torchaudio.transforms.MelSpectrogram):
    """
    torchaudio MelSpectrogram wrapper for audiomentations's Compose
    """
    def __init__(self, clip_min_value=1e-5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip_min_value = clip_min_value

    def forward(self, samples: Union[np.ndarray, torch.Tensor], sample_rate: int) -> torch.Tensor:
        if not isinstance(samples, torch.Tensor):
            samples = torch.tensor(samples)
        samples = super(MelSpectrogram, self).forward(samples)
        samples.clamp_(min=self.clip_min_value)
        return samples


class Squeeze:
    """
    Transform to squeeze monochannel waveform
    """
    def __call__(self, samples: Union[np.ndarray, torch.Tensor], sample_rate: int):
        return samples.squeeze(0)


class ToNumpy:
    """
    Transform to make numpy array
    """
    def __call__(self, samples: Union[np.ndarray, torch.Tensor], sample_rate: int):
        return np.array(samples)


class LogTransform(nn.Module):
    """
    Transform for taking logarithm of mel spectrograms (or anything else)
    :param fill_value: value to substitute non-positive numbers with before applying log
    """
    def __init__(self, fill_value: float = 1e-5) -> None:
        super().__init__()
        self.fill_value = fill_value

    def __call__(self, samples: torch.Tensor, sample_rate: int):
        samples = samples + torch.full_like(samples, self.fill_value) * (samples <= 0)
        return torch.log(samples)


class LabelEncoder:
    """
    LabelEncoder for transcripts
    :param transcripts: list of transcripts to fit to
    """
    def __init__(self, transcripts: Sequence[str]) -> None:
        self.le = torchnlp.encoders.text.CharacterEncoder({char for t in transcripts for char in list(t)})

    def __call__(self, samples: str) -> torch.Tensor:
        return self.le.encode(samples)

    @classmethod
    def from_file(cls, filename):
        le = cls([])
        le.load(filename)
        return le

    def dump(self, filename: str) -> None:
        """
        Dump encoder to disk
        :param filename: filename to dump to
        """
        with open(filename, "wb") as f:
            pickle.dump(self.le, f)

    def load(self, filename: str) -> None:
        """
        Load encoder from disk
        :param filename: filename to dump to
        """
        with open(filename, "rb") as f:
            self.le = pickle.load(f)
