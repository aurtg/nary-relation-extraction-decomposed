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

class Model(nn.Module):
    def __init__(self, n_token, n_token_text, arities, args):
        super(Model, self).__init__()

        self.args = args
        self.arities = arities

        gpus = args.gpus if args.gpu >= 0 else None
        gpus = [args.gpu] if (gpus is None) else gpus
        if (len(gpus) == 1) and (gpus[0] == -1):
            gpus = None

        # Models for binarized relations.
        self.path_encoder = Path_Encoder(n_token, args.dim_word, args.dim_rel, gpus)
        self.symb_encoder = Binarized_Relation_Encoder(arities, args.dim_rel)

        # Models for singularized relations.
        self.single_text_encoder = \
            Single_Text_Encoder(n_token_text, args.dim_word, args.dim_rel, gpus)
        self.single_symb_encoder = \
            Singularized_Relation_Encoder(arities, args.dim_rel)

        # embedding dropout
        if args.emb_dropout > 0.0:
            self.emb_dropout = nn.Dropout(p=args.emb_dropout)
        else:
            self.emb_dropout = None

        # For score aggregation.
        self.weights = nn.ParameterList([])
        for arity in self.arities:
            param = nn.Parameter(torch.FloatTensor(arity*arity).fill_(0.))
            self.weights.append(param)

    def encode(self, lst_relations):
        u"""Encode given relations.

        Args:
            lst_relations (list) -- A list of relations. See the following note.

        Return:
            out (Tensor) -- Relation representations. (batch_num x dim_emb)

        Note:
            Format of a relation is the following:
                KB relation -- ("symbol", ((i_ent1, i_ent2), i_r))
                path -- ("path", tuple_of_token_indices)
        """
        device = next(self.parameters()).device

        symb_indices = []
        symb_batch = []
        path_indices = []
        path_batch = []

        ## Prepare input for each encoder.
        for i_rel, rel in enumerate(lst_relations):
            ## KB relation
            if rel[0] == "symbol":
                symb_indices.append(i_rel)
                (i_ent1, i_ent2), i_r = rel[1]
                symb_batch.append((i_r, i_ent1, i_ent2))
            ## path
            elif rel[0] == "path":
                path_indices.append(i_rel)
                path_batch.append(rel[1])
            else:
                raise Exception("Illegal type of relation: {}".format(rel[0]))

        ## Encode relations.
        out_embs = []
        to_indices = []

        s_internal_batch = 1000
        if len(symb_batch) > 0:
            for start in range(0, len(symb_indices), s_internal_batch):
                end = start + s_internal_batch
                ifEnd = False
                if len(symb_indices) - end < 100:
                    end = len(symb_indices)
                    ifEnd = True

                if len(symb_batch[start:end]) == 0:
                    continue

                symb_embs = self.symb_encoder(symb_batch[start:end])

                out_embs.append(symb_embs)
                to_indices += symb_indices[start:end]

                if ifEnd:
                    break

        if len(path_batch) > 0:
            for start in range(0, len(path_indices), s_internal_batch):
                end = start + s_internal_batch
                ifEnd = False
                if len(path_indices) - end < 100:
                    end = len(path_indices)
                    ifEnd = True
                if len(path_batch[start:end]) == 0:
                    print(len(path_batch), len(path_indicces), start, end)
                    continue

                path_embs = self.path_encoder(path_batch[start:end])
                #print(path_embs)

                out_embs.append(path_embs)
                to_indices += path_indices[start:end]

                if ifEnd:
                    break

        out_embs = torch.cat(out_embs, dim=0)
        to_indices = torch.LongTensor(to_indices).to(device)
        out = scatter_mean(out_embs, to_indices, dim=0, dim_size=len(lst_relations))

        # Normalize output
        #out = out / torch.norm(out, dim=1, keepdim=True)

        return out

    def single_encode(self, lst_relations):
        u"""Encode given singularized relations.

        Args:
            lst_relations (list) -- A list of relations. See the following note.

        Return:
            out (Tensor) -- Relation representations. (batch_num x dim_emb)

        Note:
            Format of a relation is the following:
                KB relation -- ("symbol", (i_ent, i_r))
                text -- ("text", (tuple of token indices, tuple of entity position indices))
        """
        device = next(self.parameters()).device

        symb_indices = []
        symb_batch = []
        text_indices = []
        text_batch = []
        text_batch_positions = []

        ## Prepare input for each encoder.
        for i_rel, rel in enumerate(lst_relations):
            ## KB relation
            if rel[0] == "symbol":
                symb_indices.append(i_rel)
                i_ent, i_r = rel[1]
                symb_batch.append((i_r, i_ent))
            ## Text
            elif rel[0] == "text":
                text_indices.append(i_rel)
                text, pos = rel[1]
                text_batch.append(text)
                text_batch_positions.append(pos)
            else:
                raise Exception("Illegal type of relation: {}".format(rel[0]))

        ## Encode relations.
        out_embs = []
        to_indices = []

        s_internal_batch = 500
        if len(symb_batch) > 0:
            for start in range(0, len(symb_batch), s_internal_batch):
                end = start + s_internal_batch
                ifEnd = False
                if len(symb_batch) - end < 100:
                    end = len(symb_batch)
                    ifEnd = True

                symb_embs = self.single_symb_encoder(symb_batch[start:end])

                out_embs.append(symb_embs)
                to_indices += symb_indices[start:end]

                if ifEnd:
                    break

        if len(text_batch) > 0:
            for start in range(0, len(text_batch), s_internal_batch):
                end = start + s_internal_batch
                ifEnd = False
                if len(text_batch) - end < 100:
                    end = len(text_batch)
                    ifEnd = True

                text_embs = self.single_text_encoder(text_batch[start:end],
                    text_batch_positions[start:end])

                out_embs.append(text_embs)
                to_indices += text_indices[start:end]

                if ifEnd:
                    break

        out_embs = torch.cat(out_embs, dim=0)
        to_indices = torch.LongTensor(to_indices).to(device)
        out = scatter_mean(out_embs, to_indices, dim=0, dim_size=len(lst_relations))

        return out

    def aggregate(self, tensor_rel_embs, query_rel):
        u"""
        Args:
            tensor_rel_embs (Tensor) -- (n_rel x emb_dim)
            query_rel (Tensor) -- 1-dim Tensor of size emb_dim.
        Return:
            out (Tensor) -- (emb_dim) 1-dim
        """
        mode = self.args.aggregation
        if mode == "max":
            out, _ = torch.max(tensor_rel_embs, dim=0)
        elif mode == "mean":
            out = torch.mean(tensor_rel_embs, dim=0)
        elif mode == "max-rel":
            score = torch.sum(tensor_rel_embs * query_rel.unsqueeze(0), dim=1)
            out = tensor_rel_embs[torch.argmax(score)]
        elif mode == "attention":
            score = torch.sum(tensor_rel_embs * query_rel.unsqueeze(0), dim=1)
            weight = torch.softmax(score, dim=0).unsqueeze(1)
            out = torch.sum(tensor_rel_embs * weight, dim=0)
        elif mode == "scaled-attention":
            score = torch.sum(tensor_rel_embs * query_rel.unsqueeze(0), dim=1)
            score = score / query_rel.size(0)**0.5
            weight = torch.softmax(score, dim=0).unsqueeze(1)
            out = torch.sum(tensor_rel_embs * weight, dim=0)

        return out

    def aggregate_multi(self, ind_lst, pos_lst, row_rel, query_rel):
        u"""
        Args:
            ind_lst (list) -- List of indices of entity (tuple) to which each row of row_rel belongs.
            pos_lst (list) -- List of indices of batch to which each row of row_rel targets.
            row_rel (Tensor) -- (n_rel x emb_dim)
            query_rel (Tensor) -- (n_batch x emb_dim)
        Return:
            out (Tensor) -- (n_batch x emb_dim)
        """
        device = row_rel.device
        mode = self.args.aggregation

        ind_lst = torch.LongTensor(ind_lst).to(device)
        pos_lst = torch.LongTensor(pos_lst).to(device)

        if mode == "attention":
            row_query = query_rel[pos_lst]

            # Scores between each row relation and query relations.
            o = torch.sum(row_rel * row_query, dim=1, keepdim=True)
            # Calculate maximum score for each entity (tuple) to circumvent exp overflow.
            min_o = torch.min(o).item()
            m,_ = scatter_max(o-min_o, ind_lst, dim=0)
            m = m + min_o
            m = m[ind_lst] #(n_row x 1)
            # Calculate weight for each row relations.
            a = torch.exp(o - m)
            sum_a = scatter_add(a, ind_lst, dim=0)
            sum_a = sum_a[ind_lst]
            w = a / sum_a #(n_row x 1)
            # Calculate weighted mean for each batch.
            weighted_row_rel = w * row_rel
            out = scatter_add(weighted_row_rel, ind_lst, dim=0)
            #(n_batch x emb_dim)
        elif mode == "scaled-attention":
            row_query = query_rel[pos_lst]

            # Scores between each row relation and query relations.
            o = torch.sum(row_rel * row_query, dim=1, keepdim=True)
            o = o / row_rel.size(1)**0.5 #(n_row x 1)
            # Calculate maximum score for each entity (tuple) to circumvent exp overflow.
            min_o = torch.min(o).item()
            m,_ = scatter_max(o-min_o, ind_lst, dim=0)
            m = m + min_o
            m = m[ind_lst] #(n_row x 1)
            # Calculate weight for each row relations.
            a = torch.exp(o - m)
            sum_a = scatter_add(a, ind_lst, dim=0)
            sum_a = sum_a[ind_lst]
            w = a / sum_a #(n_row x 1)
            # Calculate weighted mean for each batch.
            weighted_row_rel = w * row_rel
            out = scatter_add(weighted_row_rel, ind_lst, dim=0)
            #(n_batch x emb_dim)

        return out

    def aggregate_score(self, i_rels, scores):
        u"""
        Args:
            i_rels (list) -- A list of indices of relation.
            scores (Tensor) -- 2-dim Tensor (n_tuple x (arity*(arity-1))).
                Score of a sub-relation among i-th and j-th (i!=j) argument of original
                relation is stored in a flattened order.
                e.g.
                    For ternary relation, each element of this tensor corresponds:
                    [(0,0),(0,1),(1,0),(1,2),(2,0),(2,1)]-th sub-relation.
        """
        device = next(self.parameters()).device

        weighted_score = []
        for i in range(len(i_rels)):
            i_rel = i_rels[i]
            score = scores[i]

            weight = torch.softmax(self.weights[i_rel], dim=0)

            score = torch.sum(weight * score, dim=0, keepdim=True)

            weighted_score.append(score)

        weighted_score = torch.cat(weighted_score, dim=0)

        return weighted_score

    def apply_word_vectors(self, word_vectors, tokens, tokens_text):
        n_exist = 0
        with torch.no_grad():
            for i in range(1, len(tokens)): # Ignore token '<PAD>'.
                token = tokens[i]
                if token in word_vectors:
                    n_exist += 1

                    self.path_encoder.embedding.weight[i] = \
                        torch.FloatTensor(word_vectors[token])
            for i in range(1, len(tokens_text)):
                token = tokens_text[i]
                if token in word_vectors:
                    self.single_text_encoder.embedding.weight[i] = \
                        torch.FloatTensor(word_vectors[token])

    def fix_aggregation_weight(self):
        with torch.no_grad():
            for i_r in range(len(self.weights)):
                self.weights[i_r].fill_(0.0)

    def step(self):
        if self.args.fix_weight:
            self.fix_aggregation_weight()

def train(model, optimizer, facts, set_facts, tup_columns, lst_tups,
    single_facts, set_single_facts, ent_contexts, lst_ents,
    batch_size=50, K=200, M=10, wa_data=None, args=None, evaluator=None):
    u"""
    Args:
        facts (list) -- List of facts. Each fact is a tuple of (entity tuple, relation).
        set_facts (set) -- `set(facts)`
        tup_columns (dict) -- It maps each entity tuple (key) to a list of relations (value)
            which are observed with it.
        lst_tups (list) -- List of observed entity tuples.

    Note:
        All entity tuples used in this function are decomposed 'binary' tuples.
    """
    def sample_negative_tuples(fact):
        u"""Sample relations of negative facts"""
        tup, rel = fact
        n_tup = len(lst_tups)

        tups = []
        while len(tups) < K:
            i_t = np.random.choice(n_tup)
            new_tup = lst_tups[i_t]

            if (new_tup, rel) in set_facts:
                continue
            else:
                tups.append(new_tup)

        return tups

    def sample_negative_entities(fact):
        ent, rel = fact
        n_ent = len(lst_ents)

        ents = []
        while len(ents) < K:
            i_t = np.random.choice(n_ent)
            new_ent = lst_ents[i_t]

            if (new_ent, rel) in set_single_facts:
                continue
            else:
                ents.append(new_ent)

        return ents

    model.train()

    device = next(model.parameters()).device

    print(len(facts), len(single_facts))

    n_binary_batch = len(facts) // args.bs
    n_single_batch = len(single_facts) // args.bs_ent
    n_batch = max(n_binary_batch, n_single_batch)
    # Use same number of batches for all setting, in order to fix number of parameter updates.

    print("Binary/single original n_batch:")
    print(n_binary_batch, n_single_batch)

    facts = shuffle(facts)
    single_facts = shuffle(single_facts)

    if args.weight_aggregation:
        train_tuples, train_positive_tuples, set_train_positive_tuples = wa_data

        train_positive_tuples = shuffle(train_positive_tuples)

        s_wa_batch = args.wa_bs
        n_wa_batch = int(np.ceil(len(train_positive_tuples) / s_wa_batch))
        print("N n_batch: {}".format(n_wa_batch))

    print("batch size: {}/{}\tbatch num: {}".format(args.bs, args.bs_ent, n_batch))

    time_start = time.time()
    for i_global_batch in range(n_batch):
        if i_global_batch > 0:
            time_per_batch = (time.time() - time_start) / i_global_batch
        else:
            time_per_batch = "N/A"
        sys.stdout.write("Processing Batch {}/{} (t.p.b. {} sec)\r".format(i_global_batch, n_batch, time_per_batch))
        sys.stdout.flush()

        ### Evaluation
        if args.mid_eval > 1:
            if (i_global_batch+1) % int(np.ceil(n_batch / args.mid_eval)) == 0:
                evaluator.eval(model)
        model.train()

        ### Start batch process.
        optimizer.zero_grad()

        loss = torch.sum(torch.FloatTensor([0.0]).to(device))

        ###
        ### Loss of decomposed binary sub-relation.
        ###
        if not args.disableB:
            i_batch = i_global_batch % n_binary_batch

            start = i_batch * args.bs
            end = (i_batch + 1) * args.bs

            cur_batch_size = len(facts[start:end])

            ## Encode relations in facts.
            pos_rels = [rel for tup, rel in facts[start:end]]
            emb_pos_rels = model.encode(pos_rels) # (batch_size x emb_dim)

            ## Encode entity tuples (1 positive and K negative).
            # Each entity tuple is encoded using its M observed relations.
            row_rels = []
            to_lst = []
            to_ind = 0
            pos_lst = []
            idx_ranges = {}
            for i_batch, (fact_tup, fact_rel) in enumerate(facts[start:end]):
                tups = [fact_tup] + sample_negative_tuples((fact_tup, fact_rel))
                for i_tup, tup in enumerate(tups):
                    rels = tup_columns[tup]
                    # Remove positive relation from V(r) (note that r is positive tuple).
                    rels = [_ for _ in rels if _ != fact_rel]
                    if len(rels) == 0:
                        rels = [fact_rel]
                    if len(rels) > M:
                        _inds = np.random.choice(len(rels), size=M)
                        rels = [rels[_ind] for _ind in _inds]
                    idx_ranges[i_batch, i_tup] = (len(row_rels), len(row_rels)+len(rels))

                    for _rel in rels:
                        row_rels.append(_rel)
                        to_lst.append(to_ind)
                        pos_lst.append(i_batch)
                    to_ind += 1

            # TIME0 = time.time()
            emb_row_rels = model.encode(row_rels) # (n_rel, dim_emb)
            # print("Time to encode entity tuple: {}".format(time.time()-TIME0))

            # Aggregate relation embedding to get entity tuple representation.
            emb_row_rels = model.aggregate_multi(to_lst, pos_lst, emb_row_rels, emb_pos_rels)
            emb_row_rels = emb_row_rels.view(len(facts[start:end]), K+1, emb_row_rels.size(1))

            ## Calculate scores
            if model.emb_dropout is not None:
                emb_row_rels = model.emb_dropout(emb_row_rels)
            scores = torch.sum(
                emb_pos_rels.unsqueeze(1) * emb_row_rels,
                dim = 2
            ) # (batch_size x K+1)

            ## Compute negative log-likelihood loss.
            _loss = - scores[:,0] + torch.logsumexp(scores, dim=1)
            _loss = torch.mean(_loss)
            loss = loss + _loss

        ###
        ### Loss of decomposed single sub-relation.
        ###
        if not args.disableU:
            i_batch = i_global_batch % n_single_batch

            start = i_batch * args.bs_ent
            end = (i_batch + 1) * args.bs_ent

            cur_batch_size = len(single_facts[start:end])

            ## Encode unary relations in facts.
            pos_ctxs = [ctx for ent, ctx in single_facts[start:end]]
            emb_pos_ctxs = model.single_encode(pos_ctxs) # (batch_size x emb_dim)

            ## Encode entities (1 positive and K negatives)
            # Each entity is encoded using its M observed contexts.
            row_ctxs = []
            to_lst = []
            to_ind = 0
            pos_lst = []
            idx_ranges = {}
            for i_batch, (fact_ent, fact_ctx) in enumerate(single_facts[start:end]):
                ents = [fact_ent] + sample_negative_entities((fact_ent, fact_ctx))
                for i_ent, ent in enumerate(ents):
                    ctxs = ent_contexts[ent]
                    ctxs = [_ for _ in ctxs if _ != fact_ctx]
                    if len(ctxs) == 0:
                        ctxs = [fact_ctx]
                    if len(ctxs) > M:
                        _inds = np.random.choice(len(ctxs), size=M)
                        ctxs = [ctxs[_ind] for _ind in _inds]
                    idx_ranges[i_batch, i_ent] = (len(row_ctxs), len(row_ctxs)+len(ctxs))

                    row_ctxs += ctxs
                    to_lst += [to_ind] * len(ctxs)
                    pos_lst += [i_batch] * len(ctxs)
                    to_ind += 1

            # TIME0 = time.time()
            emb_row_ctxs = model.single_encode(row_ctxs)
            # print("Time to encode entities: {}".format(time.time() - TIME0))

            # Aggregate relation embedding to get entity representation.
            emb_row_ctxs = model.aggregate_multi(to_lst, pos_lst, emb_row_ctxs, emb_pos_ctxs)
            emb_row_ctxs = emb_row_ctxs.view(len(single_facts[start:end]), K+1, emb_row_ctxs.size(1))
            # (batch_size x K+1 x emb_dim)

            ## Calculate scores
            if model.emb_dropout is not None:
                emb_row_ctxs = model.emb_dropout(emb_row_ctxs)
            scores = torch.sum(
                emb_pos_ctxs.unsqueeze(1) * emb_row_ctxs,
                dim = 2
            ) # (batch_size x K+1)

            ## Compute negative log-likelihood loss.
            single_loss = - scores[:,0] + torch.logsumexp(scores, dim=1)
            loss = loss + torch.mean(single_loss)
            #loss = torch.mean(single_loss)

        ###
        ### Loss of aggregated rank for target KB relations.
        ###
        if args.weight_aggregation:
            i_wa_batch = i_global_batch % n_wa_batch
            pos_tuples = train_positive_tuples[s_wa_batch*i_wa_batch : s_wa_batch*(i_wa_batch+1)]

            if len(pos_tuples) == 0:
                continue

            weights = []

            pos_rels = [] # List of positive relations.
            row_rels = [] # List of all relations to be aggregated.
            rel_aggr_pos_idxs = [] # Mapping: row_aggr_idx -> pos_rel_idx
            rel_pos_idxs = [] # Mapping: row_rel_idx -> pos_rel_idx
            rel_aggr_idxs = [] # Mapping: row_rel_idx -> aggregated_rel_idx
            rel_weight_idxs = [] # Mapping: aggregated_rel_idx -> weight_idx
            rel_tup_idxs = [] # Mapping: aggregated_rel_idx -> tup_idx (pos1, neg1, pos2, neg2, ...)

            pos_ctxs = []
            row_ctxs = []
            ctx_aggr_pos_idxs = []
            ctx_pos_idxs = []
            ctx_aggr_idxs = []
            ctx_weight_idxs = []
            ctx_tup_idxs = []

            weight_idx = 0
            aggr_row_idx = 0
            aggr_ctx_idx = 0
            pos_rel_idx = 0
            pos_ctx_idx = 0
            for i_tup, (tup, i_r) in enumerate(pos_tuples):
                arity = model.arities[i_r]

                # Sample negative tuple.
                while True:
                    neg_tup = train_tuples[np.random.choice(len(train_tuples))]
                    if len(neg_tup) != arity:
                        continue
                    if (neg_tup, i_r) in set_train_positive_tuples:
                        continue
                    else:
                        break

                # Aggregation weights.
                weights.append(torch.softmax(model.weights[i_r], dim=0)) # (arity*arity, )

                # Positive relation (context)
                pos_rels += [
                    ("symbol", ((i_ent1, i_ent2), i_r))
                    for i_ent1 in range(arity) for i_ent2 in range(arity)
                    if i_ent1 != i_ent2
                ]
                pos_ctxs += [
                    ("symbol", (i_ent, i_r)) for i_ent in range(arity)
                ]

                # Entity tuples
                for i_ent1 in range(arity):
                    for i_ent2 in range(arity):
                        if i_ent1 == i_ent2:
                            continue

                        pos_subtup = (tup[i_ent1], tup[i_ent2])
                        neg_subtup = (neg_tup[i_ent1], neg_tup[i_ent2])

                        for i_subtup, subtup in enumerate([pos_subtup, neg_subtup]):
                            rels = tup_columns[subtup]
                            rels = [_ for _ in rels if _ != ("symbol", ((i_ent1, i_ent2), i_r))]
                            if len(rels) == 0:
                                rels = [("symbol", ((i_ent1, i_ent2), i_r))]
                            elif len(rels) > M:
                                _inds = np.random.choice(len(rels), size=M)
                                rels = [rels[_ind] for _ind in _inds]

                            row_rels += rels

                            rel_pos_idxs += [pos_rel_idx] * len(rels)
                            rel_aggr_idxs += [aggr_row_idx] * len(rels)
                            rel_weight_idxs += [weight_idx]
                            rel_tup_idxs += [i_tup*2+i_subtup]
                            rel_aggr_pos_idxs += [pos_rel_idx]

                            # Loop
                            aggr_row_idx += 1

                        # Loop
                        pos_rel_idx += 1
                        weight_idx += 1

                # Entities
                for i_ent in range(arity):
                    pos_ent = tup[i_ent]
                    neg_ent = neg_tup[i_ent]

                    for i_subent, ent in enumerate([pos_ent, neg_ent]):
                        ctxs = ent_contexts[ent]
                        ctxs = [_ for _ in ctxs if _ != ("symbol", (i_ent, i_r))]
                        if len(ctxs) == 0:
                            ctxs = [("symbol", (i_ent, i_r))]
                        elif len(ctxs) > M:
                            _inds = np.random.choice(len(ctxs), size=M)
                            ctxs = [ctxs[_ind] for _ind in _inds]

                        row_ctxs += ctxs

                        ctx_pos_idxs += [pos_ctx_idx] * len(ctxs)
                        ctx_aggr_idxs += [aggr_ctx_idx] * len(ctxs)
                        ctx_weight_idxs += [weight_idx]
                        ctx_tup_idxs += [i_tup*2+i_subent]
                        ctx_aggr_pos_idxs += [pos_ctx_idx]

                        # Loop
                        aggr_ctx_idx += 1

                    # Loop
                    pos_ctx_idx += 1
                    weight_idx += 1

            emb_pos_rels = model.encode(pos_rels)
            emb_row_rels = model.encode(row_rels)
            emb_pos_ctxs = model.single_encode(pos_ctxs)
            emb_row_ctxs = model.single_encode(row_ctxs)
            weights = torch.cat(weights, dim=0).unsqueeze(1)

            emb_aggr_rels = model.aggregate_multi(rel_aggr_idxs, rel_pos_idxs, emb_row_rels, emb_pos_rels)
            emb_aggr_ctxs = model.aggregate_multi(ctx_aggr_idxs, ctx_pos_idxs, emb_row_ctxs, emb_pos_ctxs)
            if model.emb_dropout is not None:
                emb_aggr_rels = model.emb_dropout(emb_aggr_rels)
                emb_aggr_ctxs = model.emb_dropout(emb_aggr_ctxs)

            _rel_aggr_pos_idxs = torch.LongTensor(rel_aggr_pos_idxs).to(device)
            score_aggr_rel = torch.sum(emb_aggr_rels * emb_pos_rels[_rel_aggr_pos_idxs], dim=1, keepdim=True)
            _rel_weight_idxs = torch.LongTensor(rel_weight_idxs).to(device)
            score_aggr_rel = score_aggr_rel * weights[_rel_weight_idxs] #(n_aggr x 1)

            _ctx_aggr_pos_idxs = torch.LongTensor(ctx_aggr_pos_idxs).to(device)
            score_aggr_ctx = torch.sum(emb_aggr_ctxs * emb_pos_ctxs[_ctx_aggr_pos_idxs], dim=1, keepdim=True)
            _ctx_weight_idxs = torch.LongTensor(ctx_weight_idxs).to(device)
            score_aggr_ctx = score_aggr_ctx * weights[_ctx_weight_idxs] #(n_aggr x 1)

            both_tup_idxs = rel_tup_idxs + ctx_tup_idxs
            score_aggr_both = torch.cat([score_aggr_rel, score_aggr_ctx], dim=0)

            _both_tup_idxs = torch.LongTensor(both_tup_idxs).to(device)
            scores = scatter_add(score_aggr_both, _both_tup_idxs, dim=0)
            scores = scores.view(len(pos_tuples), 2)

            loss_wa = torch.max(
                scores.new_zeros((scores.size(0),)),
                1. - scores[:,0] + scores[:,1]
            )
            #loss_wa = - nn.functional.logsigmoid(scores[:,0] - scores[:,1])
            loss = loss + torch.mean(loss_wa) * args.wa_weight

        # TIME0 = time.time()
        loss.backward()
        optimizer.step()
        # print("Time to step: {}".format(time.time()-TIME0))
        model.step()

def eval_MAP(model, items, tup_columns, tup_contexts, arities, rel_index, args):
    u"""
    Args:
        rel_index (dict) -- It maps a relation name (string) to its index.
    """
    logger = getLogger("main")

    model.eval()
    device = next(model.parameters()).device

    predicate_scores = [[] for _ in range(len(arities))]

    log_scores = {}
    # Calculate relation representations.
    for i_r in range(len(arities)):
        arity = arities[i_r]

        s_batch = args.eval_bs
        n_batch = int(np.ceil(len(items) / s_batch))
        keys = list(items.keys())
        for i_batch in range(n_batch):
            sys.stdout.write(" "*50+"\r")
            sys.stdout.write("{}/{}\t".format(i_batch, n_batch))

            start = s_batch * i_batch
            end = s_batch * (i_batch + 1)

            batch_keys = [ents for ents in keys[start:end] if len(ents)==arities[i_r]]
            if len(batch_keys) == 0:
                continue

            # List of labels
            lst_y = []
            for ent in batch_keys:
                if items[ent]["relation"] in rel_index:
                    y = rel_index[items[ent]["relation"]]
                else:
                    y = -1
                lst_y.append(y==i_r)

            # Encode relations.
            emb_rel = {}
            for i_ent1 in range(arity):
                for i_ent2 in range(arity):
                    if i_ent1 == i_ent2:
                        continue

                    emb_rel[i_ent1, i_ent2] = model.encode([("symbol", ((i_ent1, i_ent2), i_r))])
                    # (1 x dim_emb)

            for i_ent in range(arity):
                emb_rel[i_ent] = model.single_encode([
                    ("symbol", (i_ent, i_r))
                ]) # (1 x dim_emb)

            # Encode entity tuples.
            row_rels = []
            idx_ranges = []
            emb_pos_rels = []

            for ents in batch_keys:
                for i_ent1 in range(len(ents)):
                    for i_ent2 in range(len(ents)):
                        if i_ent1 == i_ent2:
                            continue

                        tup = (ents[i_ent1], ents[i_ent2])

                        rels = tup_columns[tup]
                        rels = [_ for _ in rels if _[0]=="path"]

                        idx_ranges.append((len(row_rels), len(row_rels)+len(rels)))
                        row_rels += rels
                        emb_pos_rels.append(emb_rel[i_ent1, i_ent2])

            sys.stdout.write("{}\t".format(len(row_rels)))
            emb_row_rels = model.encode(row_rels)
            x = []
            for (row_start, row_end), pos_rel in zip(idx_ranges, emb_pos_rels):
                x.append(
                    model.aggregate(emb_row_rels[row_start:row_end], pos_rel.view(-1)).view(1,-1)
                )
            x = torch.cat(x, dim=0).view(
                len(batch_keys), arity*(arity-1), emb_row_rels.size(1))
            emb_tups = x

            # Encode entities
            row_ctxs = []
            idx_ranges = []
            emb_pos_ctxs = []
            for ents in batch_keys:
                for i_ent in range(len(ents)):
                    ent = ents[i_ent]

                    ctxs = tup_contexts[ent]
                    ctxs = [_ for _ in ctxs if _[0]=="text"]

                    idx_ranges.append((len(row_ctxs), len(row_ctxs)+len(ctxs)))
                    row_ctxs += ctxs
                    emb_pos_ctxs.append(emb_rel[i_ent])

            sys.stdout.write("{}\r".format(len(row_ctxs)))
            sys.stdout.flush()
            emb_row_ctxs = model.single_encode(row_ctxs)
            x = []
            for (row_start, row_end), pos_ctx in zip(idx_ranges, emb_pos_ctxs):
                x.append(
                    model.aggregate(emb_row_ctxs[row_start:row_end], pos_ctx.view(-1)).view(1,-1)
                )
            x = torch.cat(x, dim=0).view(
                len(batch_keys), arity, emb_row_ctxs.size(1)
            )
            emb_ctxs = x

            #
            if args.weight_aggregation or ((not args.disableB) and (not args.disableU)):
                emb_args = torch.cat([emb_tups, emb_ctxs], dim=1)

                y = [emb_rel[i_ent1, i_ent2] for i_ent1 in range(arity) \
                    for i_ent2 in range(arity) if i_ent1 != i_ent2] +\
                    [emb_rel[i_ent] for i_ent in range(arity)]
                y = torch.cat(y, dim=0).view(1,arity*arity,-1)
            elif (not args.disableB) and args.disableU:
                emb_args = emb_tups
                y = [emb_rel[i_ent1, i_ent2] for i_ent1 in range(arity) \
                    for i_ent2 in range(arity) if i_ent1 != i_ent2]
                y = torch.cat(y, dim=0).view(1,arity*(arity-1),-1)
            elif args.disableB and (not args.disableU):
                emb_args = emb_ctxs
                y = [emb_rel[i_ent] for i_ent in range(arity)]
                y = torch.cat(y, dim=0).view(1,arity,-1)

            scores = torch.sum(emb_args * y, dim=2)
            if args.weight_aggregation:
                aggregated_scores = model.aggregate_score([i_r]*scores.size(0), scores)
            else:
                aggregated_scores = torch.sum(scores, dim=1)

            # Log
            #TODO: Implement to support disable*
            for i_ents, ents in enumerate(batch_keys):
                # i_idx = 0
                # for i_ent1 in range(arity):
                #     for i_ent2 in range(arity):
                #         if i_ent1==i_ent2:
                #             continue
                #         if ents not in log_scores:
                #             log_scores[ents] = {}
                #         log_scores[ents][i_r, i_ent1, i_ent2] = scores[i_ents, i_idx].item()
                #         i_idx += 1
                # for i_ent in range(arity):
                #     log_scores[ents][i_r, i_ent] = scores[i_ents, i_idx].item()
                #     i_idx += 1

                predicate_scores[i_r].append(
                    (aggregated_scores[i_ents].item(), np.random.uniform(), ents, lst_y[i_ents])
                )

    # Calculate mean average precisions for each relation type.
    MAPs = []
    all_precisions = []
    for i_r in range(len(arities)):
        scores = sorted(predicate_scores[i_r], reverse=True)

        n_all = 0
        n_pos = 0
        precisions = []
        for _s, _, _e, _y in scores:
            n_all += 1
            if _y:
                n_pos += 1
                precisions.append(n_pos / n_all)

        print("MAP for predicate {}: {}".format(i_r, np.mean(precisions)))
        logger.info("MAP for predicate {}: {}".format(i_r, np.mean(precisions)))
        MAPs.append(float(np.mean(precisions)))
        all_precisions += precisions

    return float(np.mean(all_precisions)), MAPs, log_scores

if __name__=="__main__":
    parser = ArgumentParser()

    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--bs", type=int, default=50)
    parser.add_argument("--bs_ent", type=int, default=25)
    parser.add_argument("--eval_bs", type=int, default=5)
    parser.add_argument("--decay", type=float, default=1e-4)

    parser.add_argument("--emb_dropout", type=float, default=0.2)

    parser.add_argument("--weight_aggregation", action="store_true")
    parser.add_argument("--wa_bs", type=int, default=25)
    parser.add_argument("--wa_weight", type=float, default=10.0)

    parser.add_argument("--K", type=int, default=5) # Number of negative samples.
    parser.add_argument("--M", type=int, default=2) # Number of relations to be aggregated.
    parser.add_argument("--aggregation", type=str, default="scaled-attention")

    parser.add_argument("--dim_word", type=int, default=300)
    parser.add_argument("--dim_rel", type=int, default=300)

    parser.add_argument("--init_wordvec", action="store_true")

    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--gpus", type=int, nargs="+")

    parser.add_argument("--suffix", type=str, default="tmp")
    parser.add_argument("--exp_number", type=int, default=0)

    parser.add_argument("--disableB", action="store_true")
    parser.add_argument("--disableU", action="store_true")

    parser.add_argument("--fix_weight", action="store_true")

    parser.add_argument("--separate_devtest", action="store_true")

    parser.add_argument("--label_ratio", type=float, default=1.0)
    parser.add_argument("--sparse_ratio", type=float, default=1.0)

    parser.add_argument("--data", type=str, default="wiki.data.json")

    parser.add_argument("--mid_eval", type=int, default=1)

    args = parser.parse_args()

    logger = getLogger("main")
    logger.setLevel(INFO)
    handler = FileHandler("logs/ExpLog_{}_{}.log".format(args.suffix, args.exp_number))
    handler.setLevel(INFO)
    logger.addHandler(handler)

    logger.info(str(args))

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

    items, tup_columns, tokens, tokens_ind, rel_index, arities, train_columns = \
        load_data_pairwise_path(args.data, given_items=given_items)

    items_text, tup_contexts, tokens_text, tokens_ind_text, _, _, train_contexts =\
        load_data_single_text(args.data, given_items=given_items)

    if not args.separate_devtest:
        train_columns = tup_columns
        train_contexts = tup_contexts

    # Prepare n-ary tuples for aggregation weight learning.
    if args.weight_aggregation:
        train_tuples = []
        train_positive_tuples = []
        for tup in items["train"]:
            train_tuples.append(tup)
            if items["train"][tup]["relation"] in rel_index:
                i_r = rel_index[items["train"][tup]["relation"]]
                train_positive_tuples.append(
                    (tup, i_r)
                )
        set_train_positive_tuples = set(train_positive_tuples)

        wa_data = (train_tuples, train_positive_tuples, set_train_positive_tuples)
    else:
        wa_data = None

    ## Facts of binarized relations
    facts = []
    lst_tups = []
    for tup, columns in train_columns.items():
        lst_tups.append(tup)
        if len(columns) == 0:
            continue
        for rel in columns:
            facts.append((tup, rel))
    set_facts = set(facts)

    ## Facts of singularized relations
    single_facts = []
    lst_ents = []
    for ent, contexts in train_contexts.items():
        lst_ents.append(ent)
        if len(contexts) == 0:
            continue
        for ctx in contexts:
            single_facts.append((ent, ctx))
    set_single_facts = set(single_facts)

    if args.init_wordvec:
        word_vectors = load_word_vector(args.dim_word)

    # Create model.
    device = "cpu" if args.gpu < 0 else "cuda:{}".format(args.gpu)

    model = Model(
        n_token = len(tokens),
        n_token_text = len(tokens_text),
        arities = arities,
        args = args
    )
    if args.init_wordvec:
        model.apply_word_vectors(word_vectors, tokens, tokens_text)
    model.to(device)

    optimizer = optim.Adam(
        [
            {"params": model.parameters(), "lr":args.lr, "weight_decay": args.decay}
        ]
    )

    # Training and evaluation.
    class Evaluator(object):
        def __init__(self, logger, items, tup_columns, tup_contexts, arities, rel_index, args):
            self.best_MAP_dev = -1.0
            self.logger = logger
            self.items = items
            self.tup_columns = tup_columns
            self.tup_contexts = tup_contexts
            self.arities = arities
            self.rel_index = rel_index
            self.args = args

        def eval(self, model):
            print("evaluating...")
            with torch.no_grad():
                print("dev")
                self.logger.info("dev")
                MAP_dev, MAPs_dev, log_score_dev = eval_MAP(model, self.items["dev"], self.tup_columns,\
                    self.tup_contexts, self.arities, self.rel_index, self.args)

            if MAP_dev > self.best_MAP_dev:
                print("new best model")
                self.logger.info("new best model: {} -> {}".format(self.best_MAP_dev, MAP_dev))
                self.best_MAP_dev = MAP_dev

                with torch.no_grad():
                    print("test")
                    self.logger.info("test")
                    MAP_test, MAPs_test, log_score_test = eval_MAP(model, self.items["test"], self.tup_columns,\
                        self.tup_contexts, self.arities, self.rel_index, self.args)
            else:
                MAP_test = -1.0
                MAPs_test = [-1.0]*4
            print("(MAP)\tdev:{}\ttest:{}".format(MAP_dev, MAP_test))
            self.logger.info("(MAP)\tdev:{}\ttest:{}".format(MAP_dev, MAP_test))
            self.logger.info("(MAPs)\t{}\t{}".format(MAPs_dev, MAPs_test))

            self.logger.info("best model dev: {}".format(self.best_MAP_dev))
    evaluator = Evaluator(logger, items, tup_columns, tup_contexts, arities, rel_index, args)

    for i_epoch in range(args.epoch):
        print("EPOCH: {}".format(i_epoch))
        logger.info("EPOCH: {}".format(i_epoch))

        print("training...")
        train(model, optimizer, facts, set_facts, train_columns, lst_tups,\
            single_facts, set_single_facts, train_contexts, lst_ents,\
            batch_size=args.bs, K=args.K, M=args.M, wa_data=wa_data, args=args,
            evaluator=evaluator)

        evaluator.eval(model)
