import sys

from argparse import Namespace, ArgumentParser
from logging import getLogger, FileHandler, INFO
import datetime
from functools import reduce
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
    def __init__(self, n_rel, arity, n_token, n_link_label, dim_embs, node_dropout, n_layers):
        super(Model, self).__init__()

        self.dim_embs = dim_embs

        # for surface pattern (document graph).
        self.dg_encoder = DocumentGraphEncoder(
            n_token, n_link_label, dim_embs, node_dropout, n_layers
        )

        self.classifier = nn.Linear(dim_embs["state"] * arity, n_rel+1)

    def normalize(self):
        pass
        # with torch.no_grad():
        #     self.tup_encoder.weight.div_(torch.norm(self.tup_encoder.weight, dim=1, keepdim=True))

    def apply_word_vectors(self, word_vectors, i2t):
        n_exist = 0
        with torch.no_grad():
            for i in range(len(indmap.i2t)):
                token = i2t[i]
                if token in word_vectors:
                    n_exist += 1

                    self.dg_encoder.emb_token.weight[i] = torch.FloatTensor(word_vectors[token])
        print("{} out of {} tokens are initialized with word vectors".format(n_exist, len(i2t)))

    def forward(self, doc_graphs):
        embs = self.encode_document_graphs(doc_graphs)

        pred_scores = self.classifier(embs)

        return pred_scores

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

def train(model, optimizer, train_doc_graphs, train_labels, args):
    device = next(model.parameters()).device

    model.train()

    s_batch = args.bs
    n_batch = len(train_doc_graphs) // s_batch

    train_doc_graphs, train_labels = shuffle(train_doc_graphs, train_labels)

    print("batch size: {}\tbatch num: {}".format(s_batch, n_batch))

    logsoftmax = nn.LogSoftmax(dim=1)

    for i_batch in range(n_batch):
        sys.stdout.write("Processing Batch {}/{}\r".format(i_batch, n_batch))
        sys.stdout.flush()

        start = i_batch * s_batch
        end = (i_batch + 1) * s_batch

        optimizer.zero_grad()

        doc_graphs = train_doc_graphs[start:end]
        pred_scores = model(doc_graphs)

        labels = train_labels[start:end]
        labels = torch.LongTensor(labels).to(device)
        dummy = torch.LongTensor(range(labels.size(0))).to(device)
        loss = - logsoftmax(pred_scores)[dummy, labels]
        loss = torch.mean(loss)

        loss.backward()
        optimizer.step()
        # if args.normalize:
        #     model.normalize()

def eval_MAP(model, items, indmap, arities, args):
    u"""Calculate MAP of each relation types."""
    logger = getLogger("main")
    n_predicate = len(arities)

    model.eval()
    device = next(model.parameters()).device

    logsoftmax = nn.LogSoftmax(dim=1)

    keys = [p for p in items.keys()]
    y_vec = [indmap.r2i[items[p]["relation"]] if items[p]["relation"] in indmap.r2i else -1 for p in keys]

    scores = []
    key_scores = []
    for i_p, p in enumerate(keys):
        sys.stdout.write("{}/{}\r".format(i_p, len(keys)))
        sys.stdout.flush()

        doc_graphs = items[p]["docs"]
        #pred_scores = torch.max(logsoftmax(model(doc_graphs))[:,1:], dim=0)[0].tolist()
        pred_scores = torch.max(model(doc_graphs)[:,1:], dim=0)[0].tolist()
        scores.append(pred_scores)
        key_scores.append((p, pred_scores))
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

    return np.mean(all_precisions), MAPs, key_scores

if __name__=="__main__":
    parser = ArgumentParser()

    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--decay", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--bs", type=int, default=8)

    parser.add_argument("--dim_state", type=int, default=300)
    parser.add_argument("--dim_node", type=int, default=100)
    parser.add_argument("--dim_link", type=int, default=10)
    parser.add_argument("--dim_word", type=int, default=300)

    parser.add_argument("--init_wordvec", action="store_true")
    parser.add_argument("--fix_wordvec", action="store_true")

    parser.add_argument("--node_dropout", type=float, default=0.2)
    parser.add_argument("--n_layers", type=int, default=3)

    #parser.add_argument("--normalize", action="store_true")

    parser.add_argument("--label_ratio", type=float, default=1.0)
    parser.add_argument("--sparse_ratio", type=float, default=1.0)

    parser.add_argument("--suffix", type=str, default="tmp")
    parser.add_argument("--exp_number", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--save_score", action="store_true")

    parser.add_argument("--data", type=str, default="wiki.data.json")

    args = parser.parse_args()

    logger = getLogger("main")
    logger.setLevel(INFO)
    handler = FileHandler("logs/ExpLog_{}_{}.log".format(args.suffix, args.exp_number))
    handler.setLevel(INFO)
    logger.addHandler(handler)

    logger.info(str(args))

    # Load Data.
    # Load data.
    if (args.label_ratio < 1.0) or (args.sparse_ratio < 1.0):
        items, predicates = load_raw_data(args.data)

    if args.label_ratio < 1.0:
        # Randomly filter out label information.
        # Iterate over all train data, and count each predicate's frequency
        print("Counting predicate frequency...")
        pred_keys = {} #predicate id -> list of keys
        for i_key, key in enumerate(items["train"]):
            if i_key % 1000 == 0:
                print("Processed {} out of {} entity tuples.".format(i_key, len(items["train"])))

            rel = items["train"][key]["relation"]
            if rel in predicates:
                i_r = predicates[rel]

                if i_r not in pred_keys:
                    pred_keys[i_r] = []
                pred_keys[i_r].append(key)

        # Choose entity tuples with relation labels
        for i_r in pred_keys:
            _n_key = len(pred_keys[i_r])
            n_key = max(1, int(_n_key*args.label_ratio))
            logger.info("Number of labels for Predicate {}: {} -> {}".format(i_r, _n_key, n_key))

            new_keys = np.random.choice(len(pred_keys[i_r]), size=n_key, replace=False)
            new_keys = [pred_keys[i_r][ind] for ind in new_keys]
            pred_keys[i_r] = set(new_keys)

        # Filter out relation labels
        print("Removing relation labels...")
        pred_remove_cnt = {}
        for i_key, key in enumerate(items["train"]):
            if i_key % 1000 == 0:
                print("Processed {} out of {} entity tuples.".format(i_key, len(items["train"])))

            rel = items["train"][key]["relation"]
            if rel in predicates:
                i_r = predicates[rel]

                if key not in pred_keys[i_r]:
                    items["train"][key]["relation"] = "N/A"

                    if i_r not in pred_remove_cnt:
                        pred_remove_cnt[i_r] = 0
                    pred_remove_cnt[i_r] += 1

        for i_r, cnt in pred_remove_cnt.items():
            logger.info("Number of removed label of Predicate {}: {}".format(i_r, cnt))

    if args.sparse_ratio < 1.0:
        # Randomly decrease number of srface patterns to one.
        # Choose entity tuples which will have multiple surface patterns.
        set_multiple_surface_keys = set([])

        for phase in ["train", "dev", "test"]:
            for i_key, key in enumerate(items[phase]):
                if len(items[phase][key]["docs"]) > 1:
                    set_multiple_surface_keys.add(key)

        n_multiple_surface_keys = int(len(set_multiple_surface_keys) * args.sparse_ratio)
        print("Number of entity tuples with multiple surface patterns: {} -> {}".format(
            len(set_multiple_surface_keys), n_multiple_surface_keys))

        tmp_lst_key = list(set_multiple_surface_keys)
        new_set_multiple_surface_keys = np.random.choice(len(tmp_lst_key),
            size=n_multiple_surface_keys, replace=False)
        new_set_multiple_surface_keys = set([
            tmp_lst_key[_] for _ in new_set_multiple_surface_keys
        ])

        # Remove surface patterns.
        for phase in ["train", "dev", "test"]:
            for i_key, key in enumerate(items[phase]):
                if key not in new_set_multiple_surface_keys:
                    remaining_pattern_idx = np.random.choice(len(items[phase][key]["docs"]))

                    items[phase][key]["docs"] = [items[phase][key]["docs"][remaining_pattern_idx]]

    if (args.label_ratio < 1.0) or (args.sparse_ratio < 1.0):
        given_items = items
    else:
        given_items = None

    items, indmap, arities = load_data_distant(args.data, given_items=given_items)
    #items, indmap, arities = load_data_distant(args.data, mask_entity=False)#DEBUG

    train_doc_graphs = []
    train_labels = []
    for tup, data in items["train"].items():
        label = indmap.r2i[data["relation"]]+1 if data["relation"] in indmap.r2i else 0

        for doc_graph in data["docs"]:
            train_doc_graphs.append(doc_graph)
            train_labels.append(label)

    if args.init_wordvec:
        word_vectors = load_word_vector(args.dim_word)

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
        arity = arities[0],
        n_token = len(indmap.i2t),
        n_link_label = len(indmap.i2e),
        dim_embs = dim_embs,
        node_dropout = args.node_dropout,
        n_layers = args.n_layers
    )
    if args.init_wordvec:
        model.apply_word_vectors(word_vectors, indmap.i2t)
    model.to(device)

    parameters = []
    for name, param in model.named_parameters():
        if args.fix_wordvec:
            if "dg_encoder.emb_token" in name:
                continue
        parameters.append(param)
    optimizer = optim.Adam(parameters, lr = args.lr, weight_decay = args.decay)

    best_MAP_dev = -1.0
    best_key_scores_dev = None
    for i_epoch in range(args.epoch):
        print("EPOCH: {}".format(i_epoch))
        logger.info("EPOCH: {}".format(i_epoch))

        print("training...")
        train(model, optimizer, train_doc_graphs, train_labels, args)

        print("evaluating...")
        with torch.no_grad():
            print("train")
            eval_MAP(model, items["train"], indmap, arities, args)
            print("dev")
            logger.info("dev")
            MAP_dev, MAPs_dev, key_scores_dev = eval_MAP(model, items["dev"], indmap, arities, args)

        if MAP_dev > best_MAP_dev:
            print("new best model")
            logger.info("new best model: {} -> {}".format(best_MAP_dev, MAP_dev))
            best_MAP_dev = MAP_dev
            best_key_scores_dev = key_scores_dev

            with torch.no_grad():
                print("test")
                logger.info("test")
                MAP_test, MAPs_test, _ = eval_MAP(model, items["test"], indmap, arities, args)
        else:
            MAP_test = -1.0
            MAPs_test = [-1.0] * len(arities)

        print("(MAP)\tdev:{}\ttest:{}".format(MAP_dev, MAP_test))
        logger.info("(MAP)\tdev:{}\ttest:{}".format(MAP_dev, MAP_test))
        logger.info("(MAPs)\t{}\t{}".format(MAPs_dev, MAPs_test))

        logger.info("best model dev: {}".format(best_MAP_dev))

    if args.save_score:
        pic.dump(best_key_scores_dev, open("logs/Score_{}_{}.bin".format(args.suffix, args.exp_number), "wb"), -1)
