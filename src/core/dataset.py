from pathlib import Path
from typing import Tuple, Sequence

import numpy as np
import torch
import torch.utils.data as torchdata
import torchaudio


class LJSPEECH(torchdata.Dataset):
    """
    Wrapper for torchaudio.datasets.SPEECHCOMMANDS with predefined keywords
    :param root: Path to the directory where the dataset is found or downloaded.
    :param download: Whether to download the dataset if it is not found at root path. (default: False)
    :param transforms: audiomentations transform object
    :param transcript_transforms: audiomentations transform object
    :param  eos: symbol to use as EOS token (if any)
    """
    def __init__(self, root: str, download: bool = False, transforms=None,
                 transcript_transforms=None, eos=None) -> None:
        root = Path(root)
        if download and not root.exists():
            root.mkdir()
        self.eos = eos if eos is not None else ""
        self.dataset = torchaudio.datasets.LJSPEECH(root=root, download=download)
        self.transforms = transforms
        self.transcript_transforms = transcript_transforms

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        :return: waveform, normalized_transcript
        """
        (waveform, sample_rate, transcript, normalized_transcript) = self.dataset[idx]
        normalized_transcript = normalized_transcript + self.eos
        if self.transforms is not None:
            waveform = self.transforms(samples=waveform, sample_rate=sample_rate)
        if self.transcript_transforms is not None:
            normalized_transcript = self.transcript_transforms(samples=normalized_transcript)
        return waveform, normalized_transcript  # May be not waveform already

    def __len__(self) -> int:
        return len(self.dataset)

    def get_normalized_transcript(self, idx: int) -> str:
        """
        Get normalized_transcript only from the dataset
        :param idx: object index
        :return: normalized_transcript keyword_id
        """
        fileid, transcript, normalized_transcript = self.dataset._walker[idx]
        return normalized_transcript + self.eos


class RandomBySequenceLengthSampler(torchdata.Sampler):
    """
    Samples batches by bucketing them to similar lengths examples
    (Note: drops last batch)
    :param lengths: list of lengths of examples in dataset
    :param batch_size: batch size
    :param percentile: what percentile of lengths to include (e.g. 0.9 for 90% of smallest lengths)
    """
    def __init__(self, lengths: Sequence, batch_size, percentile=1.0):
        super().__init__(lengths)
        indices = np.argsort(lengths)
        indices = indices[:int(len(indices) * percentile)]
        indices = indices[:len(indices) - (len(indices) % batch_size)]
        self.batched_indices = indices.reshape(-1, batch_size)

    def __iter__(self):
        metaindices = np.random.permutation(len(self.batched_indices))
        return iter(self.batched_indices[metaindices])

    def __len__(self):
        return len(self.batched_indices)
