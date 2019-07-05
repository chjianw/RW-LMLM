import torch
import numpy as np
import torch.nn as nn

from model.utils import DEVICE, PAD


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len, learn=False):
        super(PositionalEncoding, self).__init__()

        if learn:
            self.position_encoding = nn.Embedding(max_seq_len, d_model, padding_idx=0)
        else:
            pe = torch.tensor(
                [[pos / np.power(10000, 2 * (j // 2) / d_model) for j in range(d_model)] for pos in range(max_seq_len)],
                device=DEVICE)
            pe[:, 0::2] = torch.sin(pe[:, 0::2])
            pe[:, 1::2] = torch.cos(pe[:, 1::2])

            pad_raw = torch.zeros(1, d_model, device=DEVICE)
            pe = torch.cat((pad_raw, pe))

            self.position_encoding = nn.Embedding.from_pretrained(pe)

    def forward(self, x):
        input_pos = (x != PAD).type_as(x)
        for i in range(1, x.size(1)): input_pos[:, i] += input_pos[:, i - 1]
        input_pos = input_pos.masked_fill(x == PAD, 0)

        return self.position_encoding(input_pos)


if __name__ == '__main__':
    pe = PositionalEncoding(6, 10)
    x = torch.tensor([[[1], [3], [6]], [[3], [2], [0]]])
    p = pe(x)
    print(p)
