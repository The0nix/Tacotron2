from typing import Sequence, List, Tuple, Optional

import einops as eos
import einops.layers.torch as teos
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb

import core
import core.vocoder


class Conv1dXavier(nn.Conv1d):
    """
    nn.Conv1d layer with xavier initialization
    :param gain_nonlinearity: what nonlinearity stands after the layer ("linear", "tanh", "sigmoid", etc.)
    """
    def __init__(self, *args, gain_nonlinearity: str = "linear", **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain(gain_nonlinearity))


class LinearXavier(nn.Linear):
    """
    nn.Linear layer with xavier initialization
    :param gain_nonlinearity: what nonlinearity stands after the layer ("linear", "tanh", "sigmoid", etc.)
    """
    def __init__(self, *args, gain_nonlinearity: str = "linear", **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain(gain_nonlinearity))


class EmbeddingXavier(nn.Embedding):
    """
    nn.Embedding layer with xavier initialization
    :param gain_nonlinearity: what nonlinearity stands after the layer ("linear", "tanh", "sigmoid", etc.)
    """
    def __init__(self, *args, gain_nonlinearity: str = "linear", **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain(gain_nonlinearity))


class GuidedAttentionLoss(nn.Module):
    def __init__(self, g):
        super().__init__()
        self.g = g

    def forward(self, attention_matrix):
        """
        :param attention_matrix: torch.tensor of shape (bs, seq_len, n_frames) of attention weights for each frame
        :return: tensor of shape (bs, seq_len, n_frames): attention loss
        """
        bs, seq_len, n_frames = attention_matrix.shape
        n = torch.arange(seq_len, device=attention_matrix.device).view(-1, 1)
        t = torch.arange(n_frames, device=attention_matrix.device)
        guide = 1 - torch.exp(-(n / seq_len - t / n_frames) ** 2 / (2 * self.g ** 2))
        return attention_matrix * guide


class DropOutLSTMCell(nn.Module):
    """
    LSTM cell with dropout on hidden state
    :param input_dim: Input size of LSTMCell
    :param hidden_dim: Hidden size of LSTMCell
    :param dropout: p of dropout
    :param bias: bias in LSTMCell
    """
    def __init__(self, input_dim, hidden_dim, dropout, bias=True):
        super().__init__()
        self.lstm = nn.LSTMCell(input_dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: tensor of shape (bs, input_dim)
        :param hidden: optional hidden state for lstm: tuple of two tensord of shape (bs, hidden_dim)
        :return: two tensors of shape (bs, hidden_dim): hidden_state and cell_state
        """
        hidden_state, cell_state = self.lstm(x, hidden)
        hidden_state = self.dropout(hidden_state)
        return hidden_state, cell_state


class LocationSensitiveAttention(nn.Module):
    """
    Location sensitive attention layer.
    Takes previous attention weights, encoded input, and decoder state and outputs new attention weights
    :param attention_lstm_input_dim: input dim of attention lstm layer
    must be postnet_conv_channels[-1] + encoder_lstm_dim * 2
    :param attention_lstm_dim: hidden dim of rnn layer in attention (out_channels in linear layers)
    :param attention_hidden_dim: hidden dim of attention (out_channels in linear layers)
    :param attention_kernel_size: kernel in convolution in location part
    :param attention_location_channels: number of channels in convolution in location part
    """
    def __init__(self, attention_lstm_input_dim: int, attention_lstm_dim: int,
                 attention_hidden_dim: int, attention_kernel_size: int,
                 attention_location_channels: int) -> None:
        super().__init__()
        self.attention_lstm = DropOutLSTMCell(attention_lstm_input_dim, attention_lstm_dim, dropout=0.1)
        self.location_layer = nn.Sequential(
            Conv1dXavier(2, attention_location_channels, kernel_size=attention_kernel_size,
                         padding=(attention_kernel_size - 1) // 2, bias=False, gain_nonlinearity="tanh"),
            nn.Dropout(p=0.5),
            teos.Rearrange("bs att_ch seq_len -> bs seq_len att_ch"),
            LinearXavier(attention_location_channels, attention_hidden_dim,
                         bias=False, gain_nonlinearity="tanh"),
        )
        self.query_layer = nn.Sequential(
            LinearXavier(attention_lstm_dim, attention_hidden_dim,
                         bias=False, gain_nonlinearity="tanh")
        )
        self.vtanh = nn.Sequential(
            nn.Tanh(),
            LinearXavier(attention_hidden_dim, 1, bias=False),
            teos.Rearrange("bs seq_len 1 -> bs seq_len"),
        )

    def forward(self, attention_weights_cat: torch.Tensor, memory_processed: torch.Tensor,
                attention_lstm_input: torch.Tensor, attention_lstm_hidden: Tuple[torch.Tensor, torch.Tensor],
                mask: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        :param attention_weights_cat: tensor of shape (bs, 2, seq_len) of previous attention weights and
        cumulative attention weights stacked together
        :param memory_processed: tensor of shape (bs, seq_len, attention_hidden_dim) of encoded input processed
        with attention memory layer
        :param attention_lstm_input: tensor of shape (bs, prenet_fc_dims[-1] + encoder_lstm_dim * 2):
        input of attention lstm layer. Composed of stacked glimpse and prenet output
        :param attention_lstm_hidden: tuple with two tensors of shape (n_layers, bs, hidden_size): h_0 and c_0 for LSTM
        :param mask: mask of size (bs, seq_len) for variable length sequences
        :return: tensor and tuple:
            new_attention_weights -- tensor of size (bs, seq_len) new attention weights
            (new_attention_hidden, new_attention_cell) -- tuple with two tensors of shape (n_layers, bs, hidden_size):
            h_n and c_n for LSTM
        """
        new_attention_hidden, new_attention_cell = self.attention_lstm(attention_lstm_input, attention_lstm_hidden)
        location_processed = self.location_layer(attention_weights_cat)
        query_processed = self.query_layer(new_attention_hidden)

        new_attention_weights = self.vtanh(location_processed + query_processed.unsqueeze(1) + memory_processed)
        new_attention_weights = new_attention_weights.masked_fill(~mask, float("-inf"))
        new_attention_weights = torch.softmax(new_attention_weights, dim=1)

        return new_attention_weights, (new_attention_hidden, new_attention_cell)


class TacotronDecoder(nn.Module):
    def __init__(self, lstm_input_dim, lstm_hidden_dim, encoder_dim, n_mels) -> None:
        """
        Decoder network. Takes attended input (glimpse) concatenated with prenet output and previous lstm hidden state
        and outputs new frame, probabilities that this frame is last and new lstm hidden state
        :param lstm_input_dim: Input dim of lstm. Should be attention_lstm_dim + prenet_output_dim
        :param lstm_hidden_dim: Hidden dim for lstm (duh)
        :param encoder_dim: Size of attention context vector (glimpse)
        :param n_mels: Number of channels in final mel spectrogram
        """
        super().__init__()
        self.lstm = DropOutLSTMCell(lstm_input_dim, lstm_hidden_dim, dropout=0.1)
        self.frame_projection = LinearXavier(lstm_hidden_dim + encoder_dim, n_mels)
        self.p_projection = LinearXavier(lstm_hidden_dim + encoder_dim, 1, gain_nonlinearity="sigmoid")

    def forward(self, decoder_input: torch.Tensor, glimpse: torch.Tensor,
                lstm_hidden: Tuple[torch.Tensor, torch.Tensor]) \
            -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        :param decoder_input: torch.Tensor of shape (bs, prenet_fc_dims[-1] + attention_lstm_dim):
        glimpse and attention hidden state concatenated together
        :param glimpse: attention context vectors -- tensor of shape (bs, encoder_dim)
        :param lstm_hidden: tuple with two tensors of shape (n_layers, bs, hidden_size): h_0 and c_0 for LSTM
        :return: Tuple of three objects:
            new_frame -- torch.Tensor of shape (bs, n_mels) of new spectrogram frames
            p_end -- torch.Tensor of shape (bs, 1) of logits of probabilities that this frame is last
            new_hidden_state: tuple with two tensors of shape (n_layers, bs, hidden_size): : h_n and c_n from LSTM
        """
        lstm_hidden, lstm_cell = self.lstm(decoder_input, lstm_hidden)
        lstm_output_glimpse_concat = torch.cat([lstm_hidden, glimpse], dim=1)
        p_end = self.p_projection(lstm_output_glimpse_concat)
        new_frame = self.frame_projection(lstm_output_glimpse_concat)

        return new_frame, p_end, (lstm_hidden, lstm_cell)


class Tacotron2(pl.LightningModule):
    """
    Tacotron 2 model
    Takes LabelEncoded sentences and produces a mel spectrogram
    :param num_embeddings: Number of symbols in encoding
    :param embedding_dim: Hidden size of nn.embedding in encoder
    :param encoder_conv_kernels: List of convolution kernel sizes in encoder
    :param encoder_conv_channels: List of convolution output channels in encoder
    :param encoder_lstm_dim: Hidden size of encoder LSTM (will be multiplied by two since bidirectional)
    :param attention_lstm_dim: Hidden size of RNN in LocationSensitiveAttention layer
    :param attention_hidden_dim: Hidden size of LocationSensitiveAttention layer itself
    :param attention_location_channels: Size of location vector in LocationSensitiveAttention
    :param attention_kernel_size: Kernel in convolution in location part of LocationSensitiveAttention
    :param prenet_fc_dims: List of sizes of out_features in Linear layers in prenet
    :param decoder_lstm_dim: Hidden size of 2-layer LSTM in decoder (unidirectional)
    :param postnet_conv_kernels: List of convolution kernel sizes in postnet
    :param postnet_conv_channels: List of convolution output channels in postnet (last must be equal to n_mels)
    :param n_mels: resolution of mel spectrogram to produce
    :param optimizer_lr: learning rate for Adam optimizer
    :param vocoder: optional vocoder for saving audio samples during validation
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, encoder_conv_kernels: Sequence[int],
                 encoder_conv_channels: List[int], encoder_lstm_dim: int,
                 attention_lstm_dim: int, attention_hidden_dim: int, attention_location_channels: int,
                 attention_kernel_size: int, prenet_fc_dims: List[int], decoder_lstm_dim: int,
                 postnet_conv_kernels: Sequence[int], postnet_conv_channels: List[int],
                 n_mels: int, optimizer_lr: float, vocoder: Optional[core.vocoder.Vocoder] = None) -> None:
        super().__init__()
        self.vocoder = vocoder
        self.save_hyperparameters()
        self.optimizer_lr = optimizer_lr
        assert postnet_conv_channels[-1] == n_mels, "Last convolution in postnet must " \
                                                    "have output channels equal to n_mels"
        self.n_mels = n_mels
        self.encoder_dim = encoder_conv_channels[-1]
        self.decoder_lstm_dim = decoder_lstm_dim
        self.attention_lstm_dim = attention_lstm_dim

        # ---------- Encoder ------------- #
        encoder_layers = []
        encoder_layers.append(EmbeddingXavier(num_embeddings, embedding_dim))
        encoder_layers.append(teos.Rearrange("bs seq_len channels -> bs channels seq_len"))
        for i, (in_channels, out_channels, kernel_size) in enumerate(zip([embedding_dim] + encoder_conv_channels[:-1],
                                                                     encoder_conv_channels, encoder_conv_kernels)):
            encoder_layers.append(Conv1dXavier(in_channels, out_channels, kernel_size,
                                               padding=(kernel_size - 1) // 2, gain_nonlinearity="relu"))
            nn.Dropout(p=0.5),
            encoder_layers.append(nn.BatchNorm1d(out_channels))
            encoder_layers.append(nn.ReLU())
        encoder_layers.append(teos.Rearrange("bs channels seq_len -> bs seq_len channels"))
        encoder_layers.append(nn.LSTM(encoder_conv_channels[-1], encoder_lstm_dim,
                                      bidirectional=True, batch_first=True))
        self.encoder = nn.Sequential(*encoder_layers)
        # (bs, seq_len, embedding_dim) -> (bs, seq_len, encoder_lstm_dim * 2)

        # ---------- Attention  ---------- #
        self.attention_memory_layer = LinearXavier(encoder_conv_channels[-1], attention_hidden_dim,
                                                   bias=False, gain_nonlinearity="tanh")
        self.attention = LocationSensitiveAttention(
            attention_lstm_input_dim=encoder_lstm_dim * 2 + prenet_fc_dims[-1],
            attention_lstm_dim=attention_lstm_dim,
            attention_hidden_dim=attention_hidden_dim,
            attention_kernel_size=attention_kernel_size,
            attention_location_channels=attention_location_channels
        )
        # (bs, 2, seq_len), (bs, decoder_lstm_dim), (bs, seq_len, encoder_conv_channels[-1]) -> (bs, seq_len)

        # ---------- Prenet -------------- #
        prenet_layers = []
        for i, (in_channels, out_channels) in enumerate(zip([n_mels] + prenet_fc_dims[:-1], prenet_fc_dims)):
            prenet_layers.append(LinearXavier(in_channels, out_channels, gain_nonlinearity="relu"))
            prenet_layers.append(nn.ReLU())
        self.prenet = nn.Sequential(*prenet_layers)
        # (bs, n_mels) -> (bs, prenet_fc_dims[-1])

        # ---------- Decoder ------------- #
        self.decoder = TacotronDecoder(encoder_lstm_dim * 2 + attention_lstm_dim, decoder_lstm_dim,
                                       encoder_lstm_dim * 2, n_mels)
        # (bs, encoder_conv_channels[-1] + prenet_fc_dims[-1]), (bs, n_layers, hidden_size), (bs, n_layers, hidden_size)
        # -> (bs, seq_len)

        # ---------- Postnet ------------- #
        postnet_layers = []
        for i, (in_channels, out_channels, kernel_size) in enumerate(zip([n_mels] + postnet_conv_channels[:-1],
                                                                       postnet_conv_channels, postnet_conv_kernels)):
            gain_nonlinearity = "tanh" if i != len(postnet_conv_channels) - 1 else "linear"
            postnet_layers.append(Conv1dXavier(in_channels, out_channels, kernel_size,
                                               padding=(kernel_size - 1) // 2, gain_nonlinearity=gain_nonlinearity))
            nn.Dropout(p=0.5),
            postnet_layers.append(nn.BatchNorm1d(out_channels))
            if i != len(postnet_conv_channels) - 1:
                postnet_layers.append(nn.Tanh())
        self.postnet = nn.Sequential(*postnet_layers)
        # (bs, n_mels, seq_len) -> (bs, n_mels, seq_len)

    def forward(self, transcriptions: torch.Tensor, transcription_mask: Optional[torch.Tensor] = None,
                gt_melspecs: Optional[torch.Tensor] = None, max_iter=2000)\
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates forward pass. If gt_melspecs is None, calculates inference
        :param transcriptions: tensor of size (bs, seq_len) with label encoded transcriptions
        :param transcription_mask: mask of size (bs, seq_len) for variable length sequences.
        If None, Defaults to ones
        :param gt_melspecs: tensor of size (bs, n_mel, n_frames) or None. If it is present, used for teacher
        :param max_iter: maximum number of iterations for inference
        forcing. Otherwise we perform inference
        :return: tuple of three tensors:
            result_frames -- tensor of shape (bs, n_mel, n_frames) of predictions before postnet
            result_frames_post -- tensor of shape (bs, n_mel, n_frames) of predictions after postnet
            p_end_list -- tensor of shape (bs, n_frames) of logits of probabilities that the frame is last
            result_attention_weights -- tensor of shape (bs, n_frames, seq_len) of attention weights for each frame
        """
        if transcription_mask is None:
            transcription_mask = torch.ones_like(transcriptions).bool()
        batch_size, seq_len = transcriptions.shape
        # Encode input
        input_encoded, _ = self.encoder(transcriptions)  # -> (bs, seq_len, encoder_lstm_dim * 2)

        # Values and containers
        result_frames = [torch.zeros(batch_size, self.n_mels, device=self.device)]
        p_end_list = []
        decoder_lstm_hidden = torch.zeros(batch_size, self.decoder_lstm_dim, device=self.device)
        decoder_lstm_cell = torch.zeros(batch_size, self.decoder_lstm_dim, device=self.device)
        attention_weights = torch.zeros(batch_size, seq_len, device=self.device)
        attention_memory_processed = self.attention_memory_layer(input_encoded)
        attention_weights_cum = torch.zeros(batch_size, seq_len, device=self.device)
        attention_hidden = torch.zeros(batch_size, self.attention_lstm_dim, device=self.device)
        attention_cell = torch.zeros(batch_size, self.attention_lstm_dim, device=self.device)
        glimpse = torch.zeros(batch_size, self.encoder_dim, device=self.device)
        result_attention_weights = []

        # Generate new frames for n_frames iterations or while p is not > 0.5
        if gt_melspecs is not None:
            max_iter = gt_melspecs.shape[2]
            gt_input = torch.cat([torch.zeros(batch_size, self.n_mels, 1, device=self.device), gt_melspecs], dim=2)
        for step in range(max_iter):
            # Apply prenet
            prenet_input = gt_input[:, :, step] if gt_melspecs is not None else result_frames[-1]
            prenet_output = self.prenet(prenet_input)
            prenet_output = nn.functional.dropout(prenet_output, 0.5, True)

            # Calculate attention
            attention_weights_cat = torch.cat([attention_weights.unsqueeze(1),
                                               attention_weights_cum.unsqueeze(1)], dim=1)  # Works with backward
            attention_input = torch.cat([glimpse, prenet_output], dim=1)
            attention_weights, (attention_hidden, attention_cell) = self.attention(
                attention_weights_cat=attention_weights_cat,
                memory_processed=attention_memory_processed,
                attention_lstm_input=attention_input,
                attention_lstm_hidden=(attention_hidden, attention_cell),
                mask=transcription_mask
            )
            attention_weights_cum += attention_weights
            result_attention_weights.append(attention_weights)
            # (bs, 1, seq_len) @ (bs, seq_len, encoder_lstm_dim * 2) -> (bs, encoder_lstm_dim * 2)
            glimpse = torch.bmm(attention_weights.unsqueeze(1), input_encoded).squeeze(1)

            # Decode
            decoder_input = torch.cat([glimpse, attention_hidden], dim=1)
            new_frame, p_end, (decoder_lstm_hidden, decoder_lstm_cell) = \
                self.decoder(decoder_input, glimpse, (decoder_lstm_hidden, decoder_lstm_cell))
            result_frames.append(new_frame)
            p_end_list.append(p_end)
            if gt_melspecs is None and (p_end > 0.5).all():
                break

        # Make tensors and apply postnet
        p_end_list = torch.stack(p_end_list)
        p_end_list = eos.rearrange(p_end_list, "n_frames bs 1 -> bs n_frames")
        result_frames = torch.stack(result_frames[1:])
        result_frames = eos.rearrange(result_frames, "n_frames bs channels -> bs channels n_frames")
        result_frames_post = result_frames + self.postnet(result_frames)
        result_attention_weights = torch.stack(result_attention_weights)
        result_attention_weights = eos.rearrange(result_attention_weights, "n_frames bs seq_len -> bs seq_len n_frames")

        return result_frames, result_frames_post, p_end_list, result_attention_weights

    def step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
             batch_idx: int, inference: bool) -> Tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pass batch to network, calculate losses and return total loss with gt and predicted spectrograms
        :param batch: Tuple with four tensors: spectrograms, transcriptions, spectrogram_lengths, transcription_lengths
        :param batch_idx: batch index (duh)
        :param inference: if True, don't pass gt_melspecs to forward (perform inference)
        :return: losses dict, spectrograms, result_frames, attention_weights
        """
        spectrograms, transcriptions, spectrogram_lengths, transcription_lengths = batch
        device = spectrograms.device
        batch_size, _, seq_len = spectrograms.shape
        spectrogram_mask = core.utils.lengths_to_mask(spectrogram_lengths)
        transcription_mask = core.utils.lengths_to_mask(transcription_lengths)

        # Construct ground truth for ending prediction (ones on ending frame, zeros on other)
        gt_end = torch.zeros(batch_size, seq_len, device=device)
        gt_end[torch.arange(batch_size, device=device), (spectrogram_lengths - 1).type(torch.LongTensor)] = 1

        # Apply model
        if inference:
            result_frames, result_frames_post, p_end, attention_weights = self(transcriptions, transcription_mask)
            # Make everything the same shape as spectrograms
            padding_length = spectrograms.shape[2] - result_frames.shape[2]
            padding_value = spectrograms.min()
            result_frames = torch.nn.functional.pad(result_frames, (0, padding_length), value=padding_value)
            result_frames_post = torch.nn.functional.pad(result_frames_post, (0, padding_length), value=padding_value)
            p_end = torch.nn.functional.pad(p_end, (0, padding_length), value=0)
            attention_weights = torch.nn.functional.pad(attention_weights, (0, padding_length))
        else:
            result_frames, result_frames_post, p_end, attention_weights = self(transcriptions, transcription_mask,
                                                                               gt_melspecs=spectrograms)

        # Calculate losses
        stop_pos_weight = torch.tensor([spectrogram_mask.sum(1).type(torch.FloatTensor).mean()], device=device)
        pre_loss = nn.MSELoss(reduction="none")(result_frames, spectrograms) * spectrogram_mask.unsqueeze(1)
        pre_loss = torch.mean(pre_loss.sum(dim=[1, 2]) / spectrogram_lengths)

        post_loss = nn.MSELoss(reduction="none")(result_frames_post, spectrograms) * spectrogram_mask.unsqueeze(1)
        post_loss = torch.mean(post_loss.sum(dim=[1, 2]) / spectrogram_lengths)

        stop_loss = nn.BCEWithLogitsLoss(pos_weight=stop_pos_weight, reduction="none")(p_end, gt_end) * spectrogram_mask
        stop_loss = torch.mean(stop_loss.sum(dim=1) / spectrogram_lengths)

        att_loss = GuidedAttentionLoss(g=0.2)(attention_weights)
        att_loss = att_loss * spectrogram_mask.unsqueeze(1) * transcription_mask.unsqueeze(2)
        att_loss = torch.mean(att_loss.sum(dim=[1, 2]) / spectrogram_lengths)

        loss = pre_loss + post_loss + stop_loss + att_loss
        losses = {"loss": loss, "pre_loss": pre_loss, "post_loss": post_loss, "att_loss": att_loss}

        return losses, spectrograms, result_frames_post, attention_weights

    def training_step(self, batch, batch_idx):
        losses, _, _, _ = self.step(batch, batch_idx, inference=False)

        for loss_key, loss in losses.items():
            self.log(f"train_{loss_key}", loss)
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        # Calculate losses and results in training and inference modes
        losses, spectrograms, result_frames_post, attention_weights = self.step(batch, batch_idx, inference=False)
        losses_inf, _, result_frames_post_inf, attention_weights_inf = self.step(batch, batch_idx, inference=True)

        # Log spectrograms and loss
        divider = torch.full([spectrograms.shape[0], 3, spectrograms.shape[2]], -12., device=self.device)
        stacked_images = torch.cat([spectrograms, divider, result_frames_post], dim=1)
        stacked_images_inf = torch.cat([spectrograms, divider, result_frames_post_inf], dim=1)
        self.logger.experiment.log({"Spectrograms": [wandb.Image(im) for im in stacked_images]}, commit=False)
        self.logger.experiment.log({"Spectrograms inference": [wandb.Image(im) for im in stacked_images_inf]}, commit=False)
        self.logger.experiment.log({"Attention": [wandb.Image(im) for im in attention_weights]}, commit=False)
        self.logger.experiment.log({"Attention inference": [wandb.Image(im) for im in attention_weights_inf]}, commit=False)
        for loss_key, loss in losses.items():
            self.log(f"val_{loss_key}", loss)
        for loss_key, loss in losses_inf.items():
            self.log(f"val_{loss_key}_inf", loss)
        return losses, spectrograms, result_frames_post, result_frames_post_inf

    def validation_epoch_end(self, outputs):
        _, spectrograms, result_frames_post, result_frames_post_inf = outputs[-1]

        if self.vocoder is not None:
            true_audio = self.vocoder.inference(spectrograms[:4]).detach().cpu()
            gen_audio = self.vocoder.inference(result_frames_post[:4]).detach().cpu()
            inf_audio = self.vocoder.inference(result_frames_post_inf[:4]).detach().cpu()
            true_audio = [wandb.Audio(spec, sample_rate=22050) for spec in true_audio]
            gen_audio = [wandb.Audio(spec, sample_rate=22050) for spec in gen_audio]
            inf_audio = [wandb.Audio(spec, sample_rate=22050) for spec in inf_audio]
            self.logger.experiment.log({"True audio": true_audio}, commit=False)
            self.logger.experiment.log({"Generated audio": gen_audio}, commit=False)
            self.logger.experiment.log({"Inferenced audio": inf_audio}, commit=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.optimizer_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
