import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, d_k=None, d_v=None, dropout_p=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k if d_k else d_model // h  # d_q==d_k
        self.d_v = d_v if d_v else d_model // h
        self.h = h

        self.linear_q = nn.Linear(d_model, h * self.d_k)
        self.linear_k = nn.Linear(d_model, h * self.d_k)
        self.linear_v = nn.Linear(d_model, h * self.d_v)
        self.linear_c = nn.Linear(h * self.d_v, d_model)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, query, key, value, mask=None):
        batch_size, query_size, _ = query.size()
        key_size, value_size = key.size(1), value.size(1)  # ks=vs

        query = self.linear_q(query).view(batch_size, query_size, self.h, self.d_k).transpose(1,
                                                                                              2)  # (bs*qs*h*dk)->(bs*h*qs*dk)
        key = self.linear_k(key).view(batch_size, key_size, self.h, self.d_k).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, value_size, self.h, self.d_v).transpose(1, 2)

        ### ScaledDotProductAttention
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(
            self.d_k)  ### (bs*h*qs*dk) * (bs*h*dk*ks) -> (bs*h*qs*ks)
        if mask is not None:
            mask = mask.unsqueeze(1)  # add dim h=1 Broadcasting
            scores = scores.masked_fill(mask, -1e10)  # mask=1, value=-inf #-np.inf -> -1e10
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        attn_v = torch.matmul(attn, value)  # (bs*h*qs*ks) * (bs*h*vs*dv) -> (bs*h*qs*dv)

        ### concat
        attn_v = attn_v.transpose(1, 2).contiguous().view(batch_size, query_size,
                                                          self.h * self.d_v)  ### (bs*h*qs*dv)->(bs*qs*h*dv)->(bs*qs*hdv)
        output = self.linear_c(attn_v)

        return output, attn_v
