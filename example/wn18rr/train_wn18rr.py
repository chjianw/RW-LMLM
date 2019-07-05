from torch import optim

from preproccess.datapack import DataPack
from model.lmlm import LMLM
from train.train import Train
from model.utils import init_parameters, DEVICE


def run():
    datapack = DataPack()
    train_iter, _, _ = datapack.data_iter('../../data/rw/wn18rr_id', batch_size_train=128,
                                          batch_size_test=1000, trainfile='train_50_10.csv',
                                          validfile='valid.csv',
                                          testfile='test.csv')

    model = LMLM(ent_num=datapack.n_vocab_ent, rel_num=datapack.n_vocal_rel, ent_embedding_size=100,
                 rel_embedding_size=30, hidden_size=500, n_layers=4, dropout_p=0.1, emb_dropout_p=0.2)
    model.to(DEVICE)
    init_parameters(model)

    # optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # train
    trainer = Train(model, optimizer)
    trainer.train(train_iter, n_epoch=30, print_every_batch=100, save_every_epoch=1, label_smoothing_eps=0.2,
                  model_save_path='./parameters/wn18rr/wn18rr.pt',
                  multisave=True,  # if True, it will save multiple model files suffixed with accuracy
                  log_save_path='./parameters/wn18rr.log')


if __name__ == '__main__':
    run()
