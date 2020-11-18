import numpy as np
import torch.utils.data as torchdata
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig

import core


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    core.utils.fix_seeds(cfg.common.seed)

    # Define datasets and dataloaders:
    transforms = core.utils.get_transforms(cfg.train_transforms)
    transcript_transforms = core.transforms.LabelEncoder.from_file(
        hydra.utils.to_absolute_path(cfg.preprocessing.label_encoder_filename)
    )
    dataset = core.dataset.LJSPEECH(root=hydra.utils.to_absolute_path(cfg.data.root),
                                    transforms=transforms,
                                    transcript_transforms=transcript_transforms,
                                    download=True)
    # Split data with stratification
    train_idx, val_idx = core.utils.get_split(dataset, train_size=cfg.data.train_size, random_state=cfg.common.seed)
    train_dataset = torchdata.Subset(dataset, train_idx)
    val_dataset = torchdata.Subset(dataset, val_idx)
    # Get sample weights for balancing
    collate_fn = core.utils.PadCollator(np.log(cfg.preprocessing.clip_min_value))
    # Create dataloaders
    train_dataloader = torchdata.DataLoader(train_dataset,
                                            batch_size=cfg.training.batch_size,
                                            num_workers=cfg.training.num_workers,
                                            collate_fn=collate_fn, shuffle=True)
    val_dataloader = torchdata.DataLoader(val_dataset,
                                          batch_size=cfg.training.batch_size,
                                          num_workers=cfg.training.num_workers,
                                          collate_fn=collate_fn, shuffle=False)

    # Define model
    model = core.model.Tacotron2(n_mels=cfg.preprocessing.n_mels, n_keywords=len(cfg.data.keywords),
                                 cnn_channels=cfg.model.cnn_channels,
                                 cnn_kernel_size=cfg.model.cnn_kernel_size,
                                 gru_hidden_size=cfg.model.gru_hidden_size,
                                 attention_hidden_size=cfg.model.attention_hidden_size,
                                 optimizer_lr=cfg.optimizer.lr)

    # Define logger and trainer
    wandb_logger = pl.loggers.WandbLogger(project=cfg.wandb.project)
    wandb_logger.watch(model, log="gradients", log_freq=cfg.wandb.log_freq)
    trainer = pl.Trainer(max_epochs=cfg.training.n_epochs, gpus=cfg.training.gpus,
                         logger=wandb_logger, default_root_dir="checkpoints",
                         checkpoint_callback=pl.callbacks.ModelCheckpoint(monitor="val_loss"))

    # FIT IT!
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
