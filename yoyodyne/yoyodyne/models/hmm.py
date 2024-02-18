"""Hard monotonic neural-hmm models. Wu and Cotterell 2019.
Original implementation: https://github.com/shijie-wu/neural-transducer"""

import argparse
import heapq
from typing import List, Optional, Tuple, Union, Callable, Dict

import torch
from torch import nn

from .. import data, defaults
from . import base, modules, lstm


class HardMonotonicHmm(lstm.LSTMEncoderDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_decoder(self) -> modules.lstm.HMMLSTMDecoder:
        return modules.lstm.HMMLSTMDecoder(
            pad_idx=self.pad_idx,
            start_idx=self.start_idx,
            end_idx=self.end_idx,
            decoder_input_size=self.source_encoder.output_size,  # Only pass target embedding to decoder
            num_embeddings=self.target_vocab_size,
            dropout=self.dropout,
            bidirectional=False,
            embedding_size=self.embedding_size,
            layers=self.decoder_layers,
            hidden_size=self.hidden_size,
        )

    def _get_loss_func(
        self,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Returns the actual function used to compute loss.

        Returns:
                Callable[[torch.Tensor, torch.Tensor], torch.Tensor]: configured
                        loss function.
        """
        return None

    def decode(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        teacher_forcing: bool,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decodes a sequence given the encoded input.

        Decodes until all sequences in a batch have reached [EOS] up to
        a specified length depending on the `target` args.

        Args:
                encoder_out (torch.Tensor): batch of encoded input symbols.
                encoder_mask (torch.Tensor): mask for the batch of encoded
                        input symbols.
                teacher_forcing (bool): Whether or not to decode
                        with teacher forcing.
                target (torch.Tensor, optional): target symbols;  we
                        decode up to `len(target)` symbols. If it is None, then we
                        decode up to `self.max_target_length` symbols.

        Returns:
                predictions (torch.Tensor): tensor of predictions of shape
                        seq_len x batch_size x target_vocab_size.
        """
        batch_size, src_seq_len = encoder_mask.shape
        # Initializes hidden states for decoder LSTM.
        decoder_hiddens = self.init_hiddens(batch_size, self.decoder_layers)
        # Feed in the first decoder input, as a start tag.
        # -> B x 1.
        predictions = (
            torch.tensor(
                [self.start_idx], device=self.device, dtype=torch.long
            )
            .repeat(batch_size)
            .unsqueeze(1)
        )
        if self.training:
            predictions = target  # torch.cat((predictions, target), dim=1)
        # Tracks when each sequence has decoded an EOS.
        likelihoods = None
        finished = torch.zeros(
            batch_size, device=self.device, requires_grad=False
        ).bool()
        num_steps = target.size(1) if self.training else self.max_target_length
        for t in range(num_steps):
            tgt_symbol = predictions[:, t].unsqueeze(-1)
            decoded = self.decoder(
                tgt_symbol,
                decoder_hiddens,
                encoder_out,
                encoder_mask,
            )
            transmissions_prob, emissions_prob, decoder_hiddens = (
                decoded.transitions,
                decoded.emissions,
                decoded.hiddens,
            )
            if self.training:
                likelihoods = self.score_likelihood(
                    tgt_symbol,
                    transmissions_prob,
                    emissions_prob,
                    likelihoods=likelihoods,
                )
            else:
                likelihoods, word = self.decode_step(
                    tgt_symbol,
                    transmissions_prob,
                    emissions_prob,
                    likelihoods=likelihoods,
                )
                with torch.no_grad():
                    finished |= word == self.end_idx
                    if finished.all().item():
                        break
                    word = word.where(
                        ~finished,
                        torch.tensor(self.pad_idx, device=self.device),
                    )
                    predictions = torch.cat(
                        (predictions, word.unsqueeze(-1)), dim=1
                    )

        return predictions, likelihoods

    def score_likelihood(
        self, tgt_symbol, transmissions, emissions, likelihoods=None
    ) -> Union[torch.Tensor, torch.Tensor]:
        if likelihoods is None:
            return transmissions[:, 0].unsqueeze(1) + self._get_token_prob(
                emissions, tgt_symbol
            )
        # decoder output: B x 1 x dec_hidden.
        # Make transition probs for each state
        likelihoods = likelihoods + transmissions.transpose(1, 2)
        likelihoods = likelihoods.logsumexp(dim=-1, keepdim=True).transpose(
            1, 2
        )
        likelihoods = likelihoods + self._get_token_prob(emissions, tgt_symbol)
        return likelihoods

    def decode_step(
        self, tgt_symbol, transmissions, emissions, likelihoods=None
    ) -> Union[torch.Tensor, torch.Tensor]:
        if likelihoods is None:
            likelihoods = transmissions[:, 0].unsqueeze(1)
        else:
            likelihoods = likelihoods + transmissions.transpose(1, 2)
            likelihoods = likelihoods.logsumexp(
                dim=-1, keepdim=True
            ).transpose(1, 2)

        log_probs = likelihoods + emissions.transpose(1, 2)
        log_probs = log_probs.logsumexp(dim=-1)

        word = torch.max(log_probs, dim=1)[1]
        likelihoods = likelihoods + self._get_token_prob(emissions, word)

        return likelihoods, word

    def _get_token_prob(self, probs, symb):
        bs, seq_len, _ = probs.shape
        symb = symb.view(bs, 1).expand(bs, seq_len)
        emiss = torch.gather(probs, -1, symb.unsqueeze(-1))
        mask = (symb != self.pad_idx).unsqueeze(-1).float()
        return (emiss * mask).view(bs, 1, seq_len)

    def training_step(self, batch: data.PaddedBatch, batch_idx: int) -> Dict:
        """Runs one step of training.

        This is called by the PL Trainer.

        Args:
                batch (data.PaddedBatch)
                batch_idx (int).

        Returns:
                torch.Tensor: loss.
        """
        # Forward pass produces loss by default.
        _, likelihood = self(batch)
        loss = -torch.logsumexp(likelihood, dim=-1).mean()
        self.log(
            "train_loss",
            loss,
            batch_size=len(batch),
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch: data.PaddedBatch, batch_idx: int) -> Dict:
        predictions, likelihood = self(batch)
        loss = -torch.logsumexp(likelihood, dim=-1).mean()
        # Processes for accuracy calculation.
        predictions = self.evaluator.finalize_predictions(
            predictions, self.end_idx, self.pad_idx
        )
        val_eval_item = self.evaluator.get_eval_item(
            predictions, batch.target.padded, self.pad_idx
        )
        return {"val_eval_item": val_eval_item, "val_loss": loss}

    def forward(
        self,
        batch: data.PaddedBatch,
    ) -> torch.Tensor:
        """Runs the encoder-decoder model.

        Args:
                batch (data.PaddedBatch).

        Returns:
                predictions (torch.Tensor): tensor of predictions of shape
                        (seq_len, batch_size, target_vocab_size).
        """
        encoder_out = self.source_encoder(batch.source).output
        predictions, likelihoods = self.decode(
            encoder_out,
            batch.source.mask,
            self.teacher_forcing if self.training else False,
            batch.target.padded if batch.target else None,
        )
        # -> B x seq_len x target_vocab_size.
        return predictions, likelihoods
