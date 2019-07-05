import torch
import torch.nn as nn
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"
# DEVICE='cuda:0'

PAD = 1


def get_padding_mask(query, key):
    m_q, m_k = query.unsqueeze(-1) != PAD, key.unsqueeze(-1) != PAD
    return torch.matmul(m_q.float(), m_k.transpose(-1, -2).float()) == 0


def get_sequence_mask(query):
    return torch.tensor(np.triu(np.ones((query.size(0), query.size(1), query.size(1))), k=1), dtype=torch.uint8,
                        device=DEVICE)


def init_parameters(model, from_file=None):
    if from_file:
        model.load_state_dict(torch.load(from_file, map_location=DEVICE))
    else:
        [nn.init.xavier_uniform_(p) for p in model.parameters() if p.dim() > 1]
