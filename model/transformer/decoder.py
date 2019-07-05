import torch
import torch.nn as nn
import numpy as np

from model.utils import get_padding_mask, get_sequence_mask
from model.transformer.multihead_attention import MultiHeadAttention
from model.transformer.positionwise_feedforward import PositionalWiseFeedForward
from model.transformer.positional_encoding import PositionalEncoding


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout_p=0.0):
        super(DecoderLayer, self).__init__()

        self.attention = MultiHeadAttention(d_model, n_head, dropout_p=dropout_p)
        self.feedforward = PositionalWiseFeedForward(d_model, d_ff, dropout_p)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x, self_attn_mask):
        output, self_attn = self.attention(query=x, key=x, value=x, mask=self_attn_mask)
        output = self.layernorm(x + output)
        output = self.layernorm(output + self.feedforward(output))

        return output, self_attn


class Decoder(nn.Module):
    def __init__(self, d_model, d_ff, n_head, n_layers=1, max_seq_len=10, dropout_p=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_head, d_ff, dropout_p) for _ in range(n_layers)])
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

    def forward(self, emb_x, x):
        output = emb_x * np.sqrt(self.d_model)
        output += self.pos_encoding(x)

        padding_mask = get_padding_mask(x, x)
        seq_mask = get_sequence_mask(x)
        self_attn_mask = torch.gt(padding_mask + seq_mask, 0)

        self_attentions = []
        for decoder in self.decoder_layers:
            output, self_attn = decoder(output, self_attn_mask)
            self_attentions.append(self_attn)

        return output, torch.stack(self_attentions)
