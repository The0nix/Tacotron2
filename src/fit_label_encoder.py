import hydra
from omegaconf import DictConfig

import core


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    core.utils.fix_seeds(cfg.common.seed)

    # Define dataset:
    dataset_class = core.utils.get_dataset_class(cfg.data.dataset)
    dataset = dataset_class(root=hydra.utils.to_absolute_path(cfg.data.root), eos=cfg.data.eos)

    # Split data with stratification
    train_idx, val_idx = core.utils.get_split(dataset, train_size=cfg.data.train_size, random_state=cfg.common.seed)

    # Fit label encoder on train data and save
    label_encoder = core.transforms.LabelEncoder([dataset.get_transcript(idx) for idx in train_idx])
    label_encoder.dump(hydra.utils.to_absolute_path(cfg.preprocessing.label_encoder_filename))


if __name__ == "__main__":
    main()
