from pathlib import Path
from typing import Tuple

import torch
import torch.utils.data as torchdata
import torchaudio


class LJSPEECH(torchdata.Dataset):
    """
    Wrapper for torchaudio.datasets.SPEECHCOMMANDS with predefined keywords
    :param root: Path to the directory where the dataset is found or downloaded.
    :param url: The URL to download the dataset from, or the type of the dataset to dowload. Allowed type values are
    "speech_commands_v0.01" and "speech_commands_v0.02" (default: "speech_commands_v0.02")
    :param keywords: List of keywords that will correspond to label 1
    :param download: Whether to download the dataset if it is not found at root path. (default: False)
    :param transforms: audiomentations transform object
    """
    def __init__(self, root: str, download: bool = False, transforms=None, transcript_transforms=None) -> None:
        root = Path(root)
        if download and not root.exists():
            root.mkdir()
        self.dataset = torchaudio.datasets.LJSPEECH(root=root, download=download)
        self.transforms = transforms
        self.transcript_transforms = transcript_transforms

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        :return: waveform, normalized_transcript
        """
        (waveform, sample_rate, transcript, normalized_transcript) = self.dataset[idx]
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
        return normalized_transcript
