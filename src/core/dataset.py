import csv
from pathlib import Path
from typing import Tuple, Sequence, Union, Any

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

    def get_transcript(self, idx: int) -> str:
        """
        Get normalized_transcript only from the dataset
        :param idx: object index
        :return: normalized_transcript keyword_id
        """
        fileid, transcript, normalized_transcript = self.dataset._walker[idx]
        return normalized_transcript + self.eos


class RUSLAN(torchdata.Dataset):
    """
    RUSLAN russian corpus TTS dataset. Must contain transcripts_file csv file
    and data_folder folder with 22050 sample rate files
    :param root: Path to the directory where the dataset is found or downloaded.
    :param data_folder: name of the folder with data inside root. RUSLAN by default
    :param transcripts_file: name of the file with transcripts inside root. metadata_RUSLAN_22200.csv by default
    :param transforms: audiomentations transform object
    :param transcript_transforms: audiomentations transform object
    :param eos: symbol to use as EOS token (if any)
    """
    def __init__(self, root, data_folder: str = "RUSLAN", transcripts_file: str = "metadata_RUSLAN_22200.csv",
                 transforms=None, transcript_transforms=None, eos=None) -> None:
        super().__init__()
        root_path = Path(root)
        self.filepaths = sorted(list((root_path / data_folder).glob("**/*")))
        with open(root_path / transcripts_file, "r") as f:
            reader = csv.reader(f, delimiter="|")
            self.transcripts = [tr for _, tr in sorted(reader)]

        self.transforms = transforms
        self.transcript_transforms = transcript_transforms
        self.eos = eos

    def __getitem__(self, idx: int) -> Tuple[Union[torch.Tensor, Any], Union[str, Any]]:
        """
        :return: waveform, transcript
        """
        waveform, sample_rate = torchaudio.load(self.filepaths[idx])
        transcript = self.transcripts[idx] + self.eos

        if self.transforms is not None:
            waveform = self.transforms(samples=waveform, sample_rate=sample_rate)
        if self.transcript_transforms is not None:
            transcript = self.transcript_transforms(samples=transcript)

        return waveform, transcript  # May be not waveform already

    def __len__(self) -> int:
        return len(self.filepaths)

    def get_transcript(self, idx: int) -> str:
        """
        Get transcript only from the dataset
        :param idx: object index
        :return: normalized_transcript keyword_id
        """
        return self.transcripts[idx] + self.eos


class RandomBySequenceLengthSampler(torchdata.Sampler):
    """
    Samples batches by bucketing them to similar lengths examples
    (Note: drops last batch)
    :param lengths: list of lengths of examples in dataset
    :param batch_size: batch size
    :param epoch_size: number of batches in one epoch
    :param percentile: what percentile of lengths to include (e.g. 0.9 for 90% of smallest lengths)
    """
    def __init__(self, lengths: Sequence, batch_size: int, epoch_size: int, percentile: float = 1.0):
        super().__init__(lengths)
        self.epoch_size = epoch_size
        indices = np.argsort(lengths)
        indices = indices[:int(len(indices) * percentile)]
        indices = indices[:len(indices) - (len(indices) % batch_size)]
        self.batched_indices = indices.reshape(-1, batch_size)

    def __iter__(self):
        metaindices = np.random.choice(len(self.batched_indices), size=self.epoch_size, replace=True)
        return iter(self.batched_indices[metaindices])

    def __len__(self):
        return self.epoch_size
