from collections import defaultdict
import pickle

"""
Create a graph for a triple file.
"""


class Graph():
    def __init__(self, filepath):
        self.path = filepath
        self.graph_ere = defaultdict(dict)
        self.graph_eer = defaultdict(dict)
        self.ent_set = set()
        self.rel_set = set()
        self.create_graph()
        self._degree_pd()

    def create_graph(self):
        with open(self.path) as f:
            for line in f.readlines():
                h, r, t = line.split()
                self._add_edge(h, t, r)
                self._add_edge(t, h, '-' + r)
                self.ent_set.add(h)
                self.ent_set.add(t)
                self.rel_set.add(r)
                self.rel_set.add('-' + r)

    def _degree_pd(self):       # unused
        for h in self.graph_ere:
            pd = [len(self.graph_ere[h][r]) for r in self.graph_ere[h]]
            degree = sum(pd)
            self.graph_ere[h]['d'] = degree
            self.graph_ere[h]['dpd'] = [x / degree for x in pd]

    def _add_edge(self, h, t, r):
        try:
            self.graph_eer[h][t].append(r)
        except:
            self.graph_eer[h][t] = [r]
        try:
            self.graph_ere[h][r].append(t)
        except:
            self.graph_ere[h][r] = [t]

    def filter(self, filtfile):
        with open(filtfile) as f:
            for line in f.readlines():
                h, r, t = line.split()
                self.graph_ere[h][r].remove(t)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.graph_ere, f)


if __name__ == '__main__':
    # create graph using all triples for filter setting

    all_g = Graph('../data/id/wn18rr_id/all.txt')
    all_g.save('../data/graph/wn18rr_id_all.gh')

    # all_g = Graph('../data/id/fb15k_237_id/all.txt')
    # all_g.save('../scratch/fb15k_237_id_all.gh')

    # all_g = Graph('../data/id/wn18_id/all.txt')
    # all_g.save('../scratch/wn18_id_all.gh')

    # all_g = Graph('../data/id/fb15k_id/all.txt')
    # all_g.save('../scratch/fb15k_id_all.gh')
