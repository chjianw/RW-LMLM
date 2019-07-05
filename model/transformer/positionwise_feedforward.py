import torch.nn as nn
import torch.nn.functional as F


### Linear
class PositionalWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_p=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))
