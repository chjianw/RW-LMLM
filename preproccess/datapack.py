from torchtext import data
from model.utils import DEVICE

"""
Create iterators for training, validation, and test data.
"""


class DataPack():
    def __init__(self):
        tokenize = lambda x: x.split()
        self.EP = data.Field(sequential=True, tokenize=tokenize, batch_first=True)
        self.RP = data.Field(sequential=True, tokenize=tokenize, batch_first=True)

        self.n_vocab_ent = None
        self.n_vocal_rel = None

    def data_iter(self, path, batch_size_train, batch_size_test=1, trainfile='train.csv', validfile='valid.csv',
                  testfile='test.csv'):
        train, val, test = data.TabularDataset.splits(path=path, train=trainfile, validation=validfile,
                                                      test=testfile, format='csv',
                                                      fields=[('ent_path', self.EP), ('rel_path', self.RP)])

        self.EP.build_vocab(train, val, test, min_freq=1)
        self.RP.build_vocab(train, val, test, min_freq=1)

        self.n_vocab_ent = len(self.EP.vocab)
        self.n_vocal_rel = len(self.RP.vocab)

        train_iter = data.BucketIterator(train, batch_size=batch_size_train, train=True,
                                         sort_key=lambda x: len(x.ent_path),
                                         sort=True, sort_within_batch=None, device=DEVICE)
        val_iter = data.Iterator(val, batch_size=batch_size_test, sort=False, train=False, device=DEVICE)
        test_iter = data.Iterator(test, batch_size=batch_size_test, sort=False, train=False, device=DEVICE)

        return train_iter, val_iter, test_iter
