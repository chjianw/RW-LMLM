import torch
import torch.nn.functional as F
import math
import time
import os

from model.utils import PAD, DEVICE


class Train():
    def __init__(self, model, optimizer, label_smoothing=False):
        self.model = model
        self.optimizer = optimizer
        self.smoothing = label_smoothing

    def train(self, train_iter, n_epoch, print_every_batch=10, save_every_epoch=1, model_save_path='model.pt',
              multisave=False, log_save_path='model.log', label_smoothing_eps=None):
        self.model.train()
        batch_size = train_iter.batch_size
        for e in range(n_epoch):
            loss_e, n_correct_e, n_ents_e = 0, 0, 0

            for i, batch in enumerate(train_iter):
                self.optimizer.zero_grad()
                ent_path, rel_path = batch.ent_path, batch.rel_path
                src, trg = ent_path[:, :-1], ent_path[:, 1:]
                pred = self.model(src, rel_path)
                loss, n_correct = self._calculate_loss(pred, trg, label_smoothing_eps)
                loss.div(src.size(0)).backward()  # /batch_size
                self.optimizer.step()

                loss_e += loss
                n_correct_e += n_correct
                n_ents = (trg != PAD).sum().item()
                n_ents_e += n_ents

                if i % print_every_batch == 0:
                    print('---batch %d--- loss: %f, n_correct: %d, n_ents: %d, loss_per_ent: %f, accuracy: %f'
                          % (i, loss.item(), n_correct, n_ents, loss.item() / n_ents, n_correct / n_ents))

            print('---------epoch %d-------- loss: %f, n_correct: %d, n_ents:%d, loss_per_ent: %f, accuracy: %f'
                  % (e, loss_e.item(), n_correct_e, n_ents_e, loss_e.item() / n_ents_e, n_correct_e / n_ents_e))

            if not os.path.exists(os.path.dirname(log_save_path)):
                os.makedirs(os.path.dirname(log_save_path))
            with open(log_save_path, 'a') as f:
                t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                f.write(
                    '%s, ent_emb_size: %d, rel_emb_size: %d, hid_size: %d, bat_size: %d, lr: %f, epoch: %d, loss: %f, n_correct: %d, n_ents: %d, loss_per_ent: %f, accuracy: %f\n'
                    % (
                        t, self.model.ent_embedding_size, self.model.rel_embedding_size, self.model.hidden_size,
                        batch_size,
                        self.optimizer.param_groups[0]['lr'], e, loss_e.item(), n_correct_e, n_ents_e,
                        loss_e.item() / n_ents_e, n_correct_e / n_ents_e))

            if save_every_epoch and e % save_every_epoch == 0:
                save_path = model_save_path + '_' + str(
                    round(n_correct_e / n_ents_e, 6)) if multisave else model_save_path
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                torch.save(self.model.state_dict(), save_path)

    def _calculate_loss(self, pred, trg, eps=None):
        if eps:
            n_class = pred.size(-1)
            one_hot = torch.zeros_like(pred).scatter_(2, trg.unsqueeze(-1), 1)
            y_s = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)

            loss = -(y_s * pred).sum(dim=2)
            loss = loss.masked_select(trg != PAD).sum()
        else:

            bs, ss, ws = pred.size(0), pred.size(1), pred.size(2)
            pred_ = pred.view(bs * ss, ws)  # B*S*W -> (BS)*W    #2D
            trg_ = trg.contiguous().view(-1)  # 1D
            loss = F.nll_loss(pred_, trg_, ignore_index=PAD, reduction='sum')

        ### correct num
        pred_y = pred.max(dim=2)[1]
        n_correct = (pred_y == trg).masked_select(trg != PAD).sum().item()

        return loss, n_correct
