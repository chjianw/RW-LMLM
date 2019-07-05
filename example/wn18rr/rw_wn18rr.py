from preproccess.graph import Graph
from preproccess.randow_walks import RandomWalk

"""
create graph using training triples and perform random walks on it
"""


def rw():
    train_g = Graph('../../data/id/wn18rr_id/train.txt')
    rw = RandomWalk(train_g)
    rw.walk(50, 10)
    rw.save('../../data/rw/wn18rr_id/train_50_10.csv')


if __name__ == '__main__':
    rw()
