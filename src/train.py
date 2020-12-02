import hydra
import numpy as np
import pytorch_lightning as pl
import torch.utils.data as torchdata
from omegaconf import DictConfig, OmegaConf

import core


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    core.utils.fix_seeds(cfg.common.seed)

    # Define datasets and dataloaders:
    transforms = core.utils.get_transforms(cfg.train_transforms)
    inference_transforms = core.utils.get_transforms(cfg.inference_transforms)
    label_encoder = core.transforms.LabelEncoder.from_file(
        hydra.utils.to_absolute_path(cfg.preprocessing.label_encoder_filename)
    )
    train_dataset = core.dataset.LJSPEECH(root=hydra.utils.to_absolute_path(cfg.data.root),
                                          transforms=transforms,
                                          transcript_transforms=label_encoder,
                                          eos=cfg.data.eos,
                                          download=True)
    val_dataset = core.dataset.LJSPEECH(root=hydra.utils.to_absolute_path(cfg.data.root),
                                        transforms=inference_transforms,
                                        transcript_transforms=label_encoder,
                                        eos=cfg.data.eos,
                                        download=True)
    dataset_lengths = np.array([len(train_dataset.get_normalized_transcript(idx)) for idx in range(len(train_dataset))])

    # Split data with stratification
    train_idx, val_idx = core.utils.get_split(train_dataset, train_size=cfg.data.train_size, random_state=cfg.common.seed)
    train_dataset = torchdata.Subset(train_dataset, train_idx)
    val_dataset = torchdata.Subset(val_dataset, val_idx)

    # Create sampler by transcription lengths
    train_dataset_lengths = dataset_lengths[train_idx]
    train_sampler = core.dataset.RandomBySequenceLengthSampler(train_dataset_lengths,
                                                               cfg.training.batch_size,
                                                               percentile=0.98)
    # Create dataloaders
    collate_fn = core.utils.PadCollator(np.log(cfg.preprocessing.clip_min_value), 0)
    train_dataloader = torchdata.DataLoader(train_dataset,
                                            batch_sampler=train_sampler,
                                            num_workers=cfg.training.num_workers,
                                            collate_fn=collate_fn)
    val_dataloader = torchdata.DataLoader(val_dataset,
                                          batch_size=cfg.training.batch_size,
                                          num_workers=cfg.training.num_workers,
                                          collate_fn=collate_fn, shuffle=False)

    # Define model
    vocoder = core.vocoder.Vocoder(hydra.utils.to_absolute_path(cfg.model.vocoder_checkpoint_path))
    postnet_conv_channels = OmegaConf.to_container(cfg.model.postnet_conv_channels, resolve=True)
    if "checkpoint_path" in cfg.model:
        model = core.model.Tacotron2.load_from_checkpoint(hydra.utils.to_absolute_path(cfg.model.checkpoint_path))
        model.vocoder = vocoder
    else:
        model = core.model.Tacotron2(num_embeddings=label_encoder.le.vocab_size,
                                     embedding_dim=cfg.model.embedding_dim,
                                     encoder_conv_kernels=cfg.model.encoder_conv_kernels,
                                     encoder_conv_channels=cfg.model.encoder_conv_channels,
                                     encoder_lstm_dim=cfg.model.encoder_lstm_dim,
                                     attention_lstm_dim=cfg.model.attention_lstm_dim,
                                     attention_hidden_dim=cfg.model.attention_hidden_dim,
                                     attention_location_channels=cfg.model.attention_location_channels,
                                     attention_kernel_size=cfg.model.attention_kernel_size,
                                     prenet_fc_dims=cfg.model.prenet_fc_dims,
                                     decoder_lstm_dim=cfg.model.decoder_lstm_dim,
                                     postnet_conv_kernels=cfg.model.postnet_conv_kernels,
                                     postnet_conv_channels=postnet_conv_channels,
                                     n_mels=cfg.preprocessing.n_mels,
                                     optimizer_lr=cfg.optimizer.lr,
                                     vocoder=vocoder)

    # Define logger and trainer
    wandb_logger = pl.loggers.WandbLogger(project=cfg.wandb.project)
    wandb_logger.watch(model, log="gradients", log_freq=cfg.wandb.log_freq)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss", save_last=True, save_top_k=1)
    lr_monitor_callback = pl.callbacks.LearningRateMonitor()
    trainer = pl.Trainer(max_epochs=cfg.training.n_epochs, gpus=cfg.training.gpus,
                         logger=wandb_logger, default_root_dir="checkpoints",
                         callbacks=[checkpoint_callback, lr_monitor_callback],
                         val_check_interval=cfg.training.val_check_interval,
                         gradient_clip_val=cfg.training.gradient_clip_val)

    # FIT IT!
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
