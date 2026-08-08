"""Transformer model architecture for MitoSeqGen.

This module defines an encoder-decoder transformer that is compatible with
codon-level sequence generation and obeys mitochondrial genetic-code constraints.
"""

import torch
import torch.nn as nn


class MitoSeqTransformer(nn.Module):
    def __init__(
        self,
        tgt_vocab_size: int,
        src_vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_position_embeddings: int = 1024,
    ):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_position_embeddings, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

    @property
    def device(self):
        return next(self.parameters()).device

    def _generate_square_subsequent_mask(self, size: int):
        mask = torch.triu(torch.ones((size, size), device=self.device) * float("-inf"), diagonal=1)
        return mask

    def forward(
        self,
        src,
        tgt,
        src_mask=None,
        tgt_mask=None,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
    ):
        src_positions = torch.arange(src.shape[0], device=src.device).unsqueeze(1)
        tgt_positions = torch.arange(tgt.shape[0], device=tgt.device).unsqueeze(1)
        src_emb = self.src_embedding(src) + self.position_embedding(src_positions)
        tgt_emb = self.tgt_embedding(tgt) + self.position_embedding(tgt_positions)
        memory = self.encoder(src_emb, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(tgt.size(0))
        output = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return self.output_projection(output)

    def encode(self, src, src_key_padding_mask=None):
        src_positions = torch.arange(src.shape[0], device=src.device).unsqueeze(1)
        src_emb = self.src_embedding(src) + self.position_embedding(src_positions)
        return self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

    def decode(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt_positions = torch.arange(tgt.shape[0], device=tgt.device).unsqueeze(1)
        tgt_emb = self.tgt_embedding(tgt) + self.position_embedding(tgt_positions)
        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(tgt.size(0))
        return self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
