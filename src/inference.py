from datetime import datetime
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torchaudio
from omegaconf import DictConfig

import core


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    output_dir = Path(hydra.utils.to_absolute_path(cfg.inference.inferenced_path))
    file_stem = datetime.now().strftime("%Y.%m.%d-%H:%M")
    output_audio_path = output_dir / (file_stem + ".wav")
    output_spec_path = output_dir / (file_stem + ".png")

    model = core.model.Tacotron2.load_from_checkpoint(hydra.utils.to_absolute_path(cfg.inference.checkpoint_path))
    model = model.eval().to(cfg.inference.device)
    vocoder = core.vocoder.Vocoder(hydra.utils.to_absolute_path(cfg.inference.vocoder_checkpoint_path))
    label_encoder = core.transforms.LabelEncoder.from_file(
        hydra.utils.to_absolute_path(cfg.inference.label_encoder_path)
    )

    encoded_text = label_encoder(cfg.inference.text + cfg.data.eos).to(cfg.inference.device)
    _, spectrogram, p_end, _ = model(encoded_text.unsqueeze(0))
    audio = vocoder.inference(spectrogram)

    torchaudio.save(str(output_audio_path), audio.cpu(), sample_rate=cfg.data.sample_rate)
    plt.imshow(spectrogram[0].detach().cpu())
    plt.savefig(output_spec_path)


if __name__ == "__main__":
    main()
