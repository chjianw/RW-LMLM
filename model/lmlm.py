import torch
import torch.nn as nn

from model.transformer.decoder import Decoder


class LMLM(nn.Module):
    def __init__(self, ent_num, rel_num, ent_embedding_size, rel_embedding_size, hidden_size, n_head=4, n_layers=1,
                 max_seq_len=100, dropout_p=0.1, emb_dropout_p=0):
        super(LMLM, self).__init__()

        self.ent_embedding_size = ent_embedding_size
        self.rel_embedding_size = rel_embedding_size
        self.hidden_size = hidden_size

        self.embedding_ent = nn.Embedding(ent_num, ent_embedding_size)
        self.embedding_rel = nn.Embedding(rel_num, rel_embedding_size)

        self.liner = nn.Linear(ent_embedding_size + rel_embedding_size, ent_embedding_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.dropout = nn.Dropout(emb_dropout_p)

        self.tf_decode = Decoder(ent_embedding_size + rel_embedding_size, hidden_size, n_head, n_layers, max_seq_len,
                                 dropout_p)

    def forward(self, x, r):
        e_e = self.embedding_ent(x)
        e_r = self.embedding_rel(r)
        e_er = torch.cat((e_e, e_r), dim=-1)
        e_er = self.dropout(e_er)

        output, _ = self.tf_decode(e_er, x)
        output = self.liner(output)
        output = torch.matmul(output, self.embedding_ent.weight.transpose(-1, -2))

        return self.log_softmax(output)
