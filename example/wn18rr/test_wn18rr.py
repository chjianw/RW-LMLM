import os

from preproccess.datapack import DataPack
from model.lmlm import LMLM
from train.predict import Predict
from model.utils import init_parameters, DEVICE


def run():
    datapack = DataPack()
    _, val_iter, test_iter = datapack.data_iter('../../data/rw/wn18rr_id', batch_size_train=128,
                                                batch_size_test=1000, trainfile='train_50_10.csv',
                                                validfile='valid.csv',
                                                testfile='test.csv')

    model = LMLM(ent_num=datapack.n_vocab_ent, rel_num=datapack.n_vocal_rel, ent_embedding_size=100,
                 rel_embedding_size=30, hidden_size=500, n_layers=4, dropout_p=0.1, emb_dropout_p=0.2)
    model.to(DEVICE)

    dirpath = './parameters/wn18rr/'
    filelist = sorted(os.listdir(dirpath))
    for fp in filelist:
        init_parameters(model, dirpath + fp)
        # predict
        predictor = Predict(model, '../../data/graph/wn18rr_id_all.gh')
        hits = [1, 3, 10]
        h_f_v, mr_f_v, mrr_f_v = predictor.predict(val_iter, isfilter=True, hits=hits)  # validation set
        h_f, mr_f, mrr_f = predictor.predict(test_iter, isfilter=True, hits=hits)  # testing set

        print('val --- mean rank:%f, mean rrank:%f ' % (mr_f_v, mrr_f_v), end='')
        for i, h in enumerate(hits):
            print('hits@%d:%f  ' % (h, h_f_v[i]), end='')
        print()

        print('test--- mean rank:%f, mean rrank:%f ' % (mr_f, mrr_f), end='')
        for i, h in enumerate(hits):
            print('hits@%d:%f  ' % (h, h_f[i],), end='')
        print()
        print(fp)


if __name__ == '__main__':
    run()
