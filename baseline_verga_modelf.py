import sys

from argparse import Namespace, ArgumentParser
from logging import getLogger, FileHandler, INFO
import datetime
from functools import reduce
from collections import defaultdict
from copy import deepcopy
import _pickle as pic
import time

import numpy as np
from sklearn.utils import shuffle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from torch_scatter import scatter_mean, scatter_max, scatter_add

from util import *
from model import *

import warnings
warnings.simplefilter('error')

class GS_LSTM(torch.nn.Module):
    u"""
    Graph State LSTM.
    """
    def __init__(self, dim_link_emb, dim_token_emb, dim_x, dim_h, aggr="add"):
        super(GS_LSTM, self).__init__()

        self.link_linear = nn.Sequential(
            nn.Linear(dim_link_emb + dim_token_emb, dim_x),
            nn.Tanh()
        )

        self.gate_i = nn.Sequential(
            nn.Linear(dim_x*2 + dim_h*2, dim_h),
            nn.Sigmoid()
        )
        self.gate_o = nn.Sequential(
            nn.Linear(dim_x*2 + dim_h*2, dim_h),
            nn.Sigmoid()
        )
        self.gate_f = nn.Sequential(
            nn.Linear(dim_x*2 + dim_h*2, dim_h),
            nn.Sigmoid()
        )
        self.gate_u = nn.Sequential(
            nn.Linear(dim_x*2 + dim_h*2, dim_h),
            nn.Tanh()
        )

        self.aggr = aggr


    def forward(self, h_node, c_node, e_link, e_token, i_from, i_to):
        u"""
        Args:
            h_node (FloatTensor) : Input hidden state of nodes.
            e_link (FloatTensor) : Embedding of each link. (n_link x dim_link)
            e_token (FloatTensor) : Embedding of each token. (n_token,)
            i_from (LongTensor) : Indices of source nodes of links. (n_link,)
            i_to (LongTensor) : Indices of target nodes of links. (n_link,)

        Returns:
            x, h
            x (FloatTensor) : Input for LSTM cell.
            h (FloatTensor) : Hidden state for LSTM cell.
        """
        link_x = self.link_linear(torch.cat([e_link, e_token[i_from]], dim=1))

        x_in = scatter_add(link_x, i_to, dim=0)
        x_out = scatter_add(link_x, i_from, dim=0)
        h_in = scatter_add(h_node[i_from], i_to, dim=0)
        h_out = scatter_add(h_node[i_to], i_from, dim=0)

        inp = torch.cat([x_in, x_out, h_in, h_out], dim=1)

        i = self.gate_i(inp)
        o = self.gate_o(inp)
        f = self.gate_f(inp)
        u = self.gate_u(inp)

        _c_node = f * c_node + i * u
        _h_node = o * torch.tanh(_c_node)

        return _h_node, _c_node

class DocumentGraphEncoder(nn.Module):
    def __init__(self, n_token, n_link_label, dim_embs, node_dropout, n_layers):
        super(DocumentGraphEncoder, self).__init__()

        self.dim_embs = dim_embs
        self.n_layers = n_layers

        # Embeddings
        self.emb_token = nn.Embedding(n_token, dim_embs["word"])
        nn.init.normal_(self.emb_token.weight, std=1/dim_embs["word"]**0.5)

        self.emb_link_label = nn.Embedding(n_link_label, dim_embs["link_label"])
        nn.init.normal_(self.emb_link_label.weight, std=1/dim_embs["link_label"])

        # Compress word vectors.
        self.compress = nn.Sequential(
            nn.Linear(dim_embs["word"], dim_embs["node"]),
            nn.Tanh()
        )
        self.dropout = nn.Dropout(p=node_dropout)

        # GS-LSTM module
        self.gslstm = GS_LSTM(
            dim_link_emb = dim_embs["link_label"],
            dim_token_emb = dim_embs["node"],
            dim_x = dim_embs["state"],
            dim_h = dim_embs["state"]
        )

    def forward(self, i_token, i_link, i_from, i_to):
        u"""
        Args:
            i_token (LongTensor) : Token indices of each node in the document graph.
            i_link (LongTensor) : Edge label indices of each edge in the document graph.
            i_from (LongTensor) : Start point indices of each edge.
            i_to (LongTensor) : End point indices of each edge.

        Return:
            h_node (FloatTensor) : Hidden representations of each node in given document graph.
        """
        ## Node embedding.
        word_emb = self.emb_token(i_token)

        node_emb = self.compress(word_emb)
        node_emb = self.dropout(node_emb)

        ## Edge embedding.
        edge_emb = self.emb_link_label(i_link)

        ## GS-LSTM
        # initial states (n_node x dim_state)
        h_node = node_emb.new_zeros((i_token.size(0), self.dim_embs["state"]))
        c_node = node_emb.new_zeros((i_token.size(0), self.dim_embs["state"]))

        for i_layer in range(self.n_layers):
            h_node, c_node = self.gslstm(h_node, c_node, edge_emb, node_emb, i_from, i_to)

        # h_node = word_emb.new_ones((i_token.size(0), self.dim_embs["state"]))
        # mean_word_emb = torch.mean(word_emb, dim=0)[:self.dim_embs["state"]]
        # h_node = h_node * mean_word_emb.unsqueeze(0)

        return h_node

# NOTE: dim_embs["rel"] should be arity * dim_embs["state"]
class Model(nn.Module):
    def __init__(self, n_rel, n_tup, n_token, n_link_label, dim_embs, node_dropout, n_layers, score_dropout):
        super(Model, self).__init__()

        self.dim_embs = dim_embs

        ## Encoders
        # for cannonicalized KB relations.
        self.rel_encoder = nn.Embedding(n_rel, dim_embs["rel"])
        nn.init.normal_(self.rel_encoder.weight, std=1/dim_embs["rel"]**0.5)

        # for surface pattern (document graph).
        self.dg_encoder = DocumentGraphEncoder(
            n_token, n_link_label, dim_embs, node_dropout, n_layers
        )

        ## Dropout
        self.score_dropout = nn.Dropout(p=score_dropout)

    def normalize(self):
        pass

    def apply_word_vectors(self, word_vectors, i2t):
        n_exist = 0
        with torch.no_grad():
            for i in range(len(indmap.i2t)):
                token = i2t[i]
                if token in word_vectors:
                    n_exist += 1

                    self.dg_encoder.emb_token.weight[i] = torch.FloatTensor(word_vectors[token])
        print("{} out of {} tokens are initialized with word vectors".format(n_exist, len(i2t)))

    def encode_relations(self, relations):
        device = next(self.parameters()).device

        orig_ind_symb = []
        symb_rels = []
        orig_ind_surf = []
        surf_rels = []
        for i_rel, rel in enumerate(relations):
            if isinstance(rel, tuple):
                surf_rels.append(rel)
                orig_ind_surf.append(i_rel)
            elif isinstance(rel, int):
                symb_rels.append(rel)
                orig_ind_symb.append(i_rel)

        # Encode cannonicalized KB relations.
        if len(symb_rels) > 0:
            emb_symb_rels = self.encode_symbolic_relations(symb_rels)

        # Encode surface patterns.
        if len(surf_rels) > 0:
            emb_surf_rels = self.encode_document_graphs(surf_rels)

        # Merge results.
        out = torch.zeros(len(relations), self.dim_embs["rel"]).to(device)
        if len(symb_rels) > 0:
            orig_ind_symb = torch.LongTensor(orig_ind_symb).to(device)
            out[orig_ind_symb] = emb_symb_rels
        if len(surf_rels) > 0:
            orig_ind_surf = torch.LongTensor(orig_ind_surf).to(device)
            out[orig_ind_surf] = emb_surf_rels

        return out

    def encode_symbolic_relations(self, rels):
        device = next(self.parameters()).device

        rels = torch.LongTensor(rels).to(device)
        return self.rel_encoder(rels)

    def encode_document_graphs(self, doc_graphs):
        device = next(self.parameters()).device

        arity = len(doc_graphs[0][-1])

        # Merge all document graphs into a single big graph.
        global_nodes = []
        global_edges = []
        global_i_from = []
        global_i_to = []

        entity_indices = []
        belonging_entities = []
        i_entity = 0

        for nodes, edges, i_from, i_to, pos in doc_graphs:
            node_ind_offset = len(global_nodes)

            global_nodes += list(nodes)
            global_edges += list(edges)

            global_i_from += map(lambda ind: ind+node_ind_offset, i_from)
            global_i_to += map(lambda ind: ind+node_ind_offset, i_to)

            assert len(pos) == arity, "Illegal number of entities: {}. It should be {}.".format(len(pos), arity)

            for i_ent, inds in enumerate(pos):
                entity_indices += map(lambda ind: ind+node_ind_offset, inds)
                belonging_entities += [i_entity] * len(inds)

                i_entity += 1

        # Encode merged document graph.
        global_nodes = torch.LongTensor(global_nodes).to(device)
        global_edges = torch.LongTensor(global_edges).to(device)
        global_i_from = torch.LongTensor(global_i_from).to(device)
        global_i_to = torch.LongTensor(global_i_to).to(device)

        h_node = self.dg_encoder(global_nodes, global_edges, global_i_from, global_i_to)

        # Calculate entity representations.
        entity_indices = torch.LongTensor(entity_indices).to(device)
        belonging_entities = torch.LongTensor(belonging_entities).to(device)

        ent_h_node = h_node[entity_indices]
        ent_reps = scatter_mean(ent_h_node, belonging_entities, dim=0, dim_size=arity*len(doc_graphs))
        ent_reps = ent_reps.view(len(doc_graphs), -1)

        return ent_reps

    def aggregate_multi(self, emb_tup_rels, emb_fact_rels, to_tup_ind):
        """
        Args:
            emb_tup_rels (Tensor) -- column relations. (i_cr x dim)
            emb_fact_rels (Tensor) -- relations of 1 pos / K neg example. (i_fr x K+1 x dim)
            to_tup_ind (list) -- index mapping from i_cr -> i_fr
        """
        device = next(self.parameters()).device

        mode = "scaled-attention"

        to_tup_ind = torch.LongTensor(to_tup_ind).to(device)

        if mode == "scaled-attention":
            # score: query (rel) [i_fr x K+1] \cdot column rel [i_fr x ?]
            query_rel = emb_fact_rels[to_tup_ind] # (i_cr x K+1 x dim)
            o = torch.sum(query_rel * emb_tup_rels.unsqueeze(1), dim=2)
                # (i_cr x K+1)
            # maximum score for each (i_fr x K+1)
            min_o = torch.min(o).item() # offset to make all score positive
            m, _ = scatter_max(o-min_o, to_tup_ind, dim=0) # (i_fr x K+1)
            m = m + min_o
            m = m[to_tup_ind] # (i_cr x K+1)
            # weight [i_cr x K+1]
            a = torch.exp(o - m) # (i_cr x K+1)
            sum_a = scatter_add(a, to_tup_ind, dim=0) # (i_fr x K+1)
            w = a / sum_a[to_tup_ind] # (i_cr x K+1)
            # Calculate weighted mean for each (i_fr x K+1)
            weighted_tup = w.unsqueeze(2) * emb_tup_rels.unsqueeze(1)
            out = scatter_add(weighted_tup, to_tup_ind, dim=0) #(i_fr x K+1 x dim)

        return out

def train(model, optimizer, train_tuples, train_lookup, indmap, lst_relations, tup2rels, args):
    def search_negative(tuple):
        n_rels = len(lst_relations)
        while True:
            pair, relation = tuple
            new_relation = np.random.randint(n_rels)
            new_relation = lst_relations[new_relation]
            if (pair, new_relation) in train_lookup:
                continue
            else:
                return new_relation

    model.train()

    s_batch = args.bs
    n_batch = len(train_tuples) // s_batch

    train_tuples = shuffle(train_tuples)

    print("batch size: {}\tbatch num: {}".format(s_batch, n_batch))

    logsoftmax = nn.LogSoftmax(dim=1)

    for i_batch in range(n_batch):
    #for i_batch in range(1):
        sys.stdout.write("Processing Batch {}/{}\r".format(i_batch, n_batch))
        sys.stdout.flush()

        start = i_batch * s_batch
        end = (i_batch + 1) * s_batch

        optimizer.zero_grad()

        batch_p_facts = train_tuples[start:end]

        fact_rels = []
        tup_rels = []
        to_tup_ind = []
        for i_fact, (i_tup, rel) in enumerate(batch_p_facts):
            rels = [_ for _ in tup2rels[i_tup] if _ != rel]
            if len(rels) == 0:
                rels = [rel]

            ind_rels = np.random.choice(len(rels), size=min(args.M, len(rels)), replace=False)
            rels = [rels[_] for _ in ind_rels]

            for _rel in rels:
                tup_rels.append(_rel)
                to_tup_ind.append(i_fact)

            fact_rels.append(rel)
            for k in range(args.K):
                neg_rel = search_negative((i_tup, rel))
                fact_rels.append(neg_rel)
        emb_fact_rels = model.encode_relations(fact_rels)
        emb_fact_rels = emb_fact_rels.view(-1, args.K+1, emb_fact_rels.size(1))
        emb_tup_rels = model.encode_relations(tup_rels)

        emb_tup = model.aggregate_multi(emb_tup_rels, emb_fact_rels, to_tup_ind)
            # (n_fr x K+1 x dim)
        scores = torch.sum(emb_fact_rels * model.score_dropout(emb_tup), dim=2)

        loss = -logsoftmax(scores)[:,0]
        loss = torch.mean(loss)

        loss.backward()
        optimizer.step()
        if args.normalize:
            model.normalize()

def eval_MAP(model, items, indmap, arities, tup2rels, args):
    u"""Calculate MAP of each relation types."""
    logger = getLogger("main")
    n_predicate = len(arities)

    model.eval()
    device = next(model.parameters()).device

    emb_relations = model.encode_relations(range(len(arities)))

    keys = [p for p in items.keys()]
    ent_tups = [indmap.p2i[p] for p in keys]
    y_vec = [indmap.r2i[items[p]["relation"]] if items[p]["relation"] in indmap.r2i else -1 for p in keys]

    s_batch = args.bs
    n_batch = int(np.ceil(len(ent_tups) / s_batch))
    scores = []
    for i_batch in range(n_batch):
        sys.stdout.write("{}/{}\r".format(i_batch, n_batch))
        sys.stdout.flush()

        start = i_batch * s_batch
        end = (i_batch + 1) * s_batch

        tup_rels = []
        to_tup_ind = []
        for i_tup, tup in enumerate(ent_tups[start:end]):
            rels = tup2rels[tup]

            for _rel in rels:
                tup_rels.append(_rel)
                to_tup_ind.append(i_tup)

        emb_tup_rels = model.encode_relations(tup_rels)
        emb_fact_rels = emb_relations.unsqueeze(0).repeat(\
            len(ent_tups[start:end]), 1, 1)
        emb_ent_tups = model.aggregate_multi(
            emb_tup_rels,
            emb_fact_rels,
            to_tup_ind
        ) # (n_tup, n_rel, dim)

        score = torch.sum(
            model.score_dropout(emb_ent_tups) * emb_fact_rels, dim=2
        ) # (n_ent_tups x n_relations)
        scores += score.tolist()
    scores = np.array(scores)

    all_precisions = []
    MAPs = []

    for i_r in range(n_predicate):
        score_y = sorted(
            list(zip(scores[:,i_r], np.random.uniform(size=scores.shape[0]), y_vec, keys)),
            reverse=True
        )

        n_all = 0
        n_pos = 0
        all_pos = []
        logs = []
        for score, _, y, key in score_y:
            n_all += 1
            logs.append((score, y, key))
            if y==i_r:
                n_pos += 1
                all_pos.append((n_all, n_pos))

        recalls = [_pos/n_pos for _all, _pos in all_pos]
        precisions = [_pos/_all for _all,_pos in all_pos]

        all_precisions += precisions

        print("MAP for predicate {}: {}".format(i_r, np.mean(precisions)))
        logger.info("MAP for predicate {}: {}".format(i_r, np.mean(precisions)))
        MAPs.append(np.mean(precisions))

    return np.mean(all_precisions), MAPs

if __name__=="__main__":
    parser = ArgumentParser()

    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--decay", type=float, default=1e-8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bs", type=int, default=50)

    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--M", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--dim_state", type=int, default=100)
    parser.add_argument("--dim_node", type=int, default=100)
    parser.add_argument("--dim_link", type=int, default=3)
    parser.add_argument("--dim_word", type=int, default=300)

    parser.add_argument("--init_wordvec", action="store_true")

    parser.add_argument("--node_dropout", type=float, default=0.0)
    parser.add_argument("--n_layers", type=int, default=5)

    parser.add_argument("--normalize", action="store_true")

    parser.add_argument("--label_ratio", type=float, default=1.0)
    parser.add_argument("--sparse_ratio", type=float, default=1.0)

    parser.add_argument("--suffix", type=str, default="tmp")
    parser.add_argument("--exp_number", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=-1)

    parser.add_argument("--data", type=str, default="wiki.data.json")

    args = parser.parse_args()

    logger = getLogger("main")
    logger.setLevel(INFO)
    handler = FileHandler("logs/ExpLog_{}_{}.log".format(args.suffix, args.exp_number))
    handler.setLevel(INFO)
    logger.addHandler(handler)

    # Load Data.
    items, indmap, observed_tuples, arities = load_data(args.data)

    if args.init_wordvec:
        word_vectors = load_word_vector(args.dim_word)

    # reduce label
    observed_tuples = filter_data(observed_tuples, len(arities), args.label_ratio,
        args.sparse_ratio)
    train_lookup = set(observed_tuples)

    # Set of KB relation and surface patterns (document graph)
    set_relations = set([])
    for i_p, i_r in train_lookup:
        set_relations.add(i_r)
    lst_relations = list(set_relations)

    # Mapping from tuple to cooccurring relations
    tup2rel = defaultdict(list)
    for i_p, i_r in train_lookup:
        tup2rel[i_p].append(i_r)

    #
    device = "cpu" if args.gpu < 0 else "cuda:{}".format(args.gpu)

    dim_embs = {
        "word": args.dim_word,
        "link_label": args.dim_link,
        "node": args.dim_node,
        "state": args.dim_state,
        "rel": args.dim_state * arities[0]
    }
    model = Model(
        n_rel = len(arities),
        n_tup = len(indmap.i2p),
        n_token = len(indmap.i2t),
        n_link_label = len(indmap.i2e),
        dim_embs = dim_embs,
        node_dropout = args.node_dropout,
        n_layers = args.n_layers,
        score_dropout = args.dropout
    )
    if args.init_wordvec:
        model.apply_word_vectors(word_vectors, indmap.i2t)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.decay)

    best_MAP_dev = -1.0
    for i_epoch in range(args.epoch):
        print("EPOCH: {}".format(i_epoch))
        logger.info("EPOCH: {}".format(i_epoch))

        print("training...")
        train(model, optimizer, observed_tuples, train_lookup, indmap, \
            lst_relations, tup2rel, args)

        print("evaluating...")
        with torch.no_grad():
            print("train")
            eval_MAP(model, items["train"], indmap, arities, tup2rel, args)
            print("dev")
            logger.info("dev")
            MAP_dev, MAPs_dev = eval_MAP(model, items["dev"], indmap, arities, \
                tup2rel, args)

        if MAP_dev > best_MAP_dev:
            print("new best model")
            logger.info("new best model: {} -> {}".format(best_MAP_dev, MAP_dev))
            best_MAP_dev = MAP_dev

            with torch.no_grad():
                print("test")
                logger.info("test")
                MAP_test, MAPs_test = eval_MAP(model, items["test"], indmap, \
                    arities, tup2rel, args)
        else:
            MAP_test = -1.0
            MAPs_test = [-1.0] * len(arities)

        print("(MAP)\tdev:{}\ttest:{}".format(MAP_dev, MAP_test))
        logger.info("(MAP)\tdev:{}\ttest:{}".format(MAP_dev, MAP_test))
        logger.info("(MAPs)\t{}\t{}".format(MAPs_dev, MAPs_test))

        logger.info("best model dev: {}".format(best_MAP_dev))
