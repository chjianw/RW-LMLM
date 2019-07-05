import torch
import numpy as np
import pickle

from model.utils import DEVICE


class Predict():
    def __init__(self, model, graphpath=None):
        self.model = model
        if graphpath:
            with open(graphpath, 'rb') as f:
                self.graph = pickle.load(f)

    def predict(self, test_iter, hits=[], isfilter=False):
        self.model.eval()
        with torch.no_grad():
            hits_s = np.zeros(len(hits))
            mrank_s, mrrank_s, bs_s = 0, 0, 0
            for i, batch in enumerate(test_iter):
                ent_path, rel_path = batch.ent_path, batch.rel_path
                src, trg = ent_path[:, :-1], ent_path[:, 1:]
                pred = self.model(src, rel_path)
                if isfilter:
                    pred = self._filter(src, rel_path, trg, pred, test_iter)

                hits_n, rank_s, r_rank_s = self._evaluate(trg, pred, hits)
                # print('---batch %d---, ishits: %s, rank: %d, r_rank: %s' % (i, hits_n, rank_s, r_rank_s))
                hits_s += hits_n
                mrank_s += rank_s
                mrrank_s += r_rank_s
                bs_s += batch.batch_size

            hits_p, mean_rank, mean_rrank = hits_s / bs_s, mrank_s / bs_s, mrrank_s / bs_s
            # print(hits_s,mrank_s,bs_s)
            # for i, h in enumerate(hits):
            #     print('hits@%d:%f, hits_n:%d, mean rank:%f, mean rrank:%f ' % (
            #     h, hits_p[i], hits_s[i], mean_rank, mean_rrank))
            return hits_p, mean_rank, mean_rrank

    def _evaluate(self, trg, pred, hits):
        bs, ss, ws = trg.size(0), trg.size(1), pred.size(-1)
        pred_rank = torch.argsort(pred, dim=-1, descending=True).view(bs * ss, ws).tolist()
        trg_ent = trg.contiguous().view(bs * ss).tolist()

        r = np.array(list(map(lambda x, y: y.index(x) + 1, trg_ent, pred_rank)))
        hits_n = np.array([np.sum(r <= h) for h in hits])
        rank_s = sum(r)
        r_rank_s = sum(1 / r)

        return hits_n, rank_s, r_rank_s

    def _filter(self, src, rel, trg, pred, test_iter):
        bs, ss, ws = pred.size(0), pred.size(1), pred.size(2)
        src_l = src.contiguous().view(bs * ss).tolist()
        rel_l = rel.contiguous().view(bs * ss).tolist()
        trg_l = trg.contiguous().view(bs * ss).tolist()
        pred_l = pred.contiguous().view(bs * ss, ws).tolist()

        ent_s2i = test_iter.dataset.fields['ent_path'].vocab.stoi
        ent_i2s = test_iter.dataset.fields['ent_path'].vocab.itos
        rel_i2s = test_iter.dataset.fields['rel_path'].vocab.itos

        def f(s, r, t, p):
            fl = [ent_s2i[x] for x in self.graph[ent_i2s[s]][rel_i2s[r]]]
            fl.remove(t)
            for i in fl: p[i] = float('-inf')
            return p

        return torch.tensor(list(map(f, src_l, rel_l, trg_l, pred_l)), device=DEVICE)
