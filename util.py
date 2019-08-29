import sys

from argparse import Namespace, ArgumentParser
from collections import defaultdict
import datetime
from functools import reduce
import json
import _pickle as pic

import numpy as np
from sklearn.utils import shuffle

from util_graph import extract_pairwise_shortest_paths

with open("directories.json") as h:
    CONFIG_DIR = json.load(h)

def orig_token(token):
    if token == "-LRB-":
        return "("
    elif token == "-RRB-":
        return ")"
    elif token == "''":
        return "\""
    elif token == "``":
        return "\""
    else:
        return token

def _load_data_graph(filename, pairs=None):
    u"""
    Load text file of document graph.
    Each document graph is organized by their entity tuple.
    """

    print("loading document graph file.")
    print(filename)
    with open(filename) as h:
        data = json.load(h)
    print("Number of sentences: {}".format(len(data)))

    print("start parsing data.")
    if pairs is None:
        pairs = {}
    for i_item, item in enumerate(data):
        if i_item % 5000 == 0:
            print("parsing {}/{}".format(i_item, len(data)))

        entity_ids = [ent["id"] for ent in item["entities"]]

        rel_label = item["relationLabel"]

        t = tuple(entity_ids)
        if t not in pairs:
            pairs[t] = {"docs":[item], "relation":rel_label}
        else:
            if pairs[t]["relation"] != rel_label:
                raise Exception("One entity pair assumed to have only one relation label.")
            pairs[t]["docs"].append(item)

    return pairs

def load_data_graph(filename):
    if isinstance(filename, str):
        return _load_data_graph(filename)
    elif isinstance(filename, list):
        pairs = {}
        for name in filename:
            _load_data_graph(name, pairs)
        return pairs
    else:
        raise Exception("Illegal input to load_data_graph: {}".format(type(filename)))

def load_data(data_json, tuple_type="all_index", add_self_loop=False, remove_path=None,
        min_word_freq=5, mask_entity=True, if_orig_token=False):
    u"""
    Args:
        tuple_type (str): "all_index" or "ent_index"
            If it is "all_index", indices are assigned to each entity tuple.
            If it is "ent_index", indices are assigned to each entity, and entity
            tuple is expressed as tuple of entity indices.
        remove_path (function): If remove_path(link) is True, the link will not be
            appended to data. Default is None, i.e. all links are appended to data.
    """
    with open(data_json) as h:
        DATA_CONFIG = json.load(h)

    ### Load data.
    print("Loading data...")

    items = {}
    for phase in ["train", "dev", "test"]:
        items[phase] = load_data_graph(
            DATA_CONFIG["data"].format(phase)
        )

    ### Create index map.
    print("Creating indmap...")

    indmap = Namespace()

    tokens = set([])
    entities = set([])
    links = set([])
    n_tokens = {}

    for phase in ["train", "dev", "test"]:
        for tup, pair in items[phase].items():
            # Register entities.
            for ent in tup:
                entities.add(ent)

            # Register tokens and links.
            for doc in pair["docs"]:
                placeholders = {}
                for i_ent, ent in enumerate(doc["entities"]):
                    for ind in ent["indices"]:
                        #placeholders[ind] = "<entity_{}>".format(i_ent)
                        placeholders[ind] = "<entity>"

                nodes = reduce(
                    lambda x,y: x+y,
                    map(lambda sent:sent["nodes"], doc["sentences"])
                )

                for n in nodes:
                    if mask_entity:
                        if n["index"] in placeholders:
                            n["lemma"] = placeholders[n["index"]]

                    tokens.add(n["lemma"])
                    if n["lemma"] not in n_tokens:
                        n_tokens[n["lemma"]] = 0
                    n_tokens[n["lemma"]] += 1

                    for arc in n["arcs"]:
                        links.add(arc["label"])

    indmap.i2t = ["<PAD>", "<UNK>"] + list(tokens)
    indmap.t2i = {t:i for i,t in enumerate(indmap.i2t)}
    indmap.t2n = n_tokens

    indmap.i2e = list(entities)
    indmap.e2i = {e:i for i,e in enumerate(indmap.i2e)}

    indmap.i2l = list(links) + ["self"]
    indmap.l2i = {l:i for i,l in enumerate(indmap.i2l)}

    indmap.i2p = list(set(items["train"].keys()) | set(items["dev"].keys()) | set(items["test"].keys()))
    indmap.p2i = {p:i for i,p in enumerate(indmap.i2p)}

    indmap.r2i = DATA_CONFIG["predicates"]

    # Create observed tuples.
    # Textual pattern format:
    #   ((nodes_indices, ...), (edge_indices, ...), (from_indices, ...), (to_indices, ...))
    observed_tuples = []

    for phase in ["train", "dev", "test"]:
        for i_pair, (tup, pair) in enumerate(items[phase].items()):
            if i_pair % 5000 == 0:
                print(i_pair, len(items[phase]))
            if tuple_type == "all_index":
                i_p = indmap.p2i[tup]
            elif tuple_type == "ent_index":
                i_p = tuple([indmap.e2i[e] for e in tup])
            else:
                raise NotImplementedError()

            # Symbolic relations (TRAIN ONLY)
            if phase == "train":
                if pair["relation"] in DATA_CONFIG["predicates"]:
                    observed_tuples.append(
                        (i_p, DATA_CONFIG["predicates"][pair["relation"]])
                    )

            # Textual relations.
            def textual_pattern_graph(doc):
                pat_nodes = []
                pat_edges = []
                pat_from = []
                pat_to = []
                pat_pos = []

                for ent in doc["entities"]:
                    pat_pos.append(tuple(ent["indices"]))

                nodes = reduce(
                    lambda x,y:x+y,
                    map(lambda sent:sent["nodes"], doc["sentences"])
                )

                for n in nodes:
                    # token
                    if if_orig_token:
                        pat_nodes.append(
                            orig_token(n["label"])
                        )
                    else:
                        pat_nodes.append(
                            indmap.t2i[n["lemma"]] if indmap.t2n[n["lemma"]] > min_word_freq else indmap.t2i["<UNK>"]
                        )
                    # Add self loop
                    if add_self_loop:
                        n["arcs"] += [
                            {
                                "toIndex": n["index"],
                                "label": "self"
                            }
                        ]
                    # links
                    for arc in n["arcs"]:
                        # Remove specified links.
                        if remove_path is not None:
                            if remove_path(arc["label"]):
                                continue

                        pat_edges.append(
                            indmap.l2i[arc["label"]]
                        )
                        pat_from.append(
                            n["index"]
                        )
                        pat_to.append(
                            arc["toIndex"]
                        )

                i_r = (
                    tuple(pat_nodes),
                    tuple(pat_edges),
                    tuple(pat_from),
                    tuple(pat_to),
                    tuple(pat_pos)
                )
                return i_r


            for i_doc, doc in enumerate(pair["docs"]):
                i_r = textual_pattern_graph(doc)

                observed_tuples.append(
                    (i_p, i_r)
                )

    arities = DATA_CONFIG["arities"]

    return items, indmap, observed_tuples, arities

def load_data_distant(data_json, tuple_type="all_index", add_self_loop=False, remove_path=None,
        min_word_freq=5, mask_entity=True, if_orig_token=False, given_items=None):
    u"""
    Args:
        tuple_type (str): "all_index" or "ent_index"
            If it is "all_index", indices are assigned to each entity tuple.
            If it is "ent_index", indices are assigned to each entity, and entity
            tuple is expressed as tuple of entity indices.
        remove_path (function): If remove_path(link) is True, the link will not be
            appended to data. Default is None, i.e. all links are appended to data.
    """
    with open(data_json) as h:
        DATA_CONFIG = json.load(h)

    ### Load data.
    print("Loading data...")

    if given_items is None:
        ### Load data.
        print("Loading data...")

        items = {}
        for phase in ["train", "dev", "test"]:
            items[phase] = load_data_graph(
                DATA_CONFIG["data"].format(phase)
            )
    else:
        items = given_items

    ### Create index map.
    print("Creating indmap...")

    indmap = Namespace()

    tokens = set([])
    entities = set([])
    links = set([])
    n_tokens = {}

    for phase in ["train", "dev", "test"]:
        for tup, pair in items[phase].items():
            # Register entities.
            for ent in tup:
                entities.add(ent)

            # Register tokens and links.
            for doc in pair["docs"]:
                placeholders = {}
                for i_ent, ent in enumerate(doc["entities"]):
                    for ind in ent["indices"]:
                        #placeholders[ind] = "<entity_{}>".format(i_ent)
                        placeholders[ind] = "<entity>"

                nodes = reduce(
                    lambda x,y: x+y,
                    map(lambda sent:sent["nodes"], doc["sentences"])
                )

                for n in nodes:
                    if mask_entity:
                        if n["index"] in placeholders:
                            n["lemma"] = placeholders[n["index"]]

                    tokens.add(n["lemma"])
                    if n["lemma"] not in n_tokens:
                        n_tokens[n["lemma"]] = 0
                    n_tokens[n["lemma"]] += 1

                    for arc in n["arcs"]:
                        links.add(arc["label"])

    indmap.i2t = ["<PAD>", "<UNK>"] + list(tokens)
    indmap.t2i = {t:i for i,t in enumerate(indmap.i2t)}
    indmap.t2n = n_tokens

    indmap.i2e = list(entities)
    indmap.e2i = {e:i for i,e in enumerate(indmap.i2e)}

    indmap.i2l = list(links) + ["self"]
    indmap.l2i = {l:i for i,l in enumerate(indmap.i2l)}

    indmap.i2p = list(set(items["train"].keys()) | set(items["dev"].keys()) | set(items["test"].keys()))
    indmap.p2i = {p:i for i,p in enumerate(indmap.i2p)}

    indmap.r2i = DATA_CONFIG["predicates"]

    # Create observed tuples.
    # Textual pattern format:
    #   ((nodes_indices, ...), (edge_indices, ...), (from_indices, ...), (to_indices, ...))
    for phase in ["train", "dev", "test"]:
        for i_pair, (tup, pair) in enumerate(items[phase].items()):
            # Textual relations.
            def textual_pattern_graph(doc):
                pat_nodes = []
                pat_edges = []
                pat_from = []
                pat_to = []
                pat_pos = []

                for ent in doc["entities"]:
                    pat_pos.append(tuple(ent["indices"]))

                nodes = reduce(
                    lambda x,y:x+y,
                    map(lambda sent:sent["nodes"], doc["sentences"])
                )

                for n in nodes:
                    # token
                    if if_orig_token:
                        pat_nodes.append(
                            orig_token(n["label"])
                        )
                    else:
                        pat_nodes.append(
                            indmap.t2i[n["lemma"]] if indmap.t2n[n["lemma"]] > min_word_freq else indmap.t2i["<UNK>"]
                        )
                    # Add self loop
                    if add_self_loop:
                        n["arcs"] += [
                            {
                                "toIndex": n["index"],
                                "label": "self"
                            }
                        ]
                    # links
                    for arc in n["arcs"]:
                        # Remove specified links.
                        if remove_path is not None:
                            if remove_path(arc["label"]):
                                continue

                        pat_edges.append(
                            indmap.l2i[arc["label"]]
                        )
                        pat_from.append(
                            n["index"]
                        )
                        pat_to.append(
                            arc["toIndex"]
                        )

                i_r = (
                    tuple(pat_nodes),
                    tuple(pat_edges),
                    tuple(pat_from),
                    tuple(pat_to),
                    tuple(pat_pos)
                )
                return i_r


            for i_doc, doc in enumerate(pair["docs"]):
                i_r = textual_pattern_graph(doc)

                items[phase][tup]["docs"][i_doc] = i_r

    arities = DATA_CONFIG["arities"]

    return items, indmap, arities

def load_data_distant_ub(data_json, tuple_type="all_index", add_self_loop=False, remove_path=None,
        min_word_freq=5, mask_entity=True, if_orig_token=False, given_items=None):
    u"""
    Args:
        tuple_type (str): "all_index" or "ent_index"
            If it is "all_index", indices are assigned to each entity tuple.
            If it is "ent_index", indices are assigned to each entity, and entity
            tuple is expressed as tuple of entity indices.
        remove_path (function): If remove_path(link) is True, the link will not be
            appended to data. Default is None, i.e. all links are appended to data.
    """
    with open(data_json) as h:
        DATA_CONFIG = json.load(h)

    ### Load data.
    print("Loading data...")

    if given_items is None:
        ### Load data.
        print("Loading data...")

        items = {}
        for phase in ["train", "dev", "test"]:
            items[phase] = load_data_graph(
                DATA_CONFIG["data"].format(phase)
            )
    else:
        items = given_items

    # Index mapping
    indmap = Namespace()

    indmap.i2w = ["<PAD>", "<UNK>", "<EMPTY>"]
    indmap.w2i = {word: i_word for i_word, word in enumerate(indmap.i2w)}

    indmap.i2t = ["<PAD>", "<UNK>", "<EMPTY>"]
    indmap.t2i = {token: i_token for i_token, token in enumerate(indmap.i2t)}

    indmap.r2i = DATA_CONFIG["predicates"]

    # Create observed tuples.
    # Textual pattern format:
    #   (tuple_of_unary_patterns, tuple_of_binary_patterns)
    # unary_pattern = (tuple of token indices, tuple of entity position indices)
    # binary_pattern = tuple_of_token_indices
    def textual_pattern_graph(doc):
        pair = extract_pairwise_shortest_paths(doc)
        paths = pair.get_subpaths()

        # Load binary surface patterns.
        binary_patterns = []
        for (i_ent1, i_ent2), path in paths:
            if len(path) == 0:
                path = ("<EMPTY>",)

            # Convert path into a sequence of token indices.
            for token in path:
                if token not in indmap.t2i.keys():
                    indmap.i2t.append(token)
                    indmap.t2i[token] = len(indmap.i2t) - 1
            path = tuple([indmap.t2i[token] for token in path])

            binary_patterns.append(path)

        # Load unary surface patterns.
        unary_patterns = []

        text = [node["lemma"] for sent in doc["sentences"] for node in sent["nodes"]]
        if len(text) == 0:
            text = ["<EMPTY>"]

        # Convert text into a sequence of word indices.
        for word in text:
            if word not in indmap.w2i.keys():
                indmap.i2w.append(word)
                indmap.w2i[word] = len(indmap.i2w) - 1
        text = tuple([indmap.w2i[word] for word in text])

        n_ent = len(doc["entities"])
        for i_ent in range(n_ent):
            pos = tuple(doc["entities"][i_ent]["indices"])

            unary_patterns.append((text, pos))

        return (unary_patterns, binary_patterns)


    for phase in ["train", "dev", "test"]:
        for i_pair, (tup, pair) in enumerate(items[phase].items()):
            for i_doc, doc in enumerate(pair["docs"]):
                i_r = textual_pattern_graph(doc)

                items[phase][tup]["docs"][i_doc] = i_r

    arities = DATA_CONFIG["arities"]

    return items, indmap, arities

def load_raw_data(data_json):
    with open(data_json) as h:
        DATA_CONFIG = json.load(h)

    ### Load data.
    print("Loading data...")

    items = {}
    for phase in ["train", "dev", "test"]:
        items[phase] = load_data_graph(
            DATA_CONFIG["data"].format(phase)
        )

    return items, DATA_CONFIG["predicates"]

def load_data_pairwise_path(data_json, min_word_freq=5, given_items=None):
    with open(data_json) as h:
        DATA_CONFIG = json.load(h)

    if given_items is None:
        ### Load data.
        print("Loading data...")

        items = {}
        for phase in ["train", "dev", "test"]:
            items[phase] = load_data_graph(
                DATA_CONFIG["data"].format(phase)
            )
    else:
        items = given_items

    ### extract pairwise shortest dependency paths
    print("Loading observed columns...")
    observed_columns = {} # KB relations in train data and all surface patterns.
    train_columns = {}
    tokens = set([])
    tokens_cnt = {}
    for phase in ["train", "dev", "test"]:
        print(phase)
        for i_key, key in enumerate(items[phase]):
            if i_key % 1000 == 0:
                print("Processed {} out of {} entity tuples.".format(i_key, len(items[phase])))
            #print(key)
            n_ent = len(key)

            # surface patterns
            for doc in items[phase][key]["docs"]:
                pair = extract_pairwise_shortest_paths(doc)
                paths = pair.get_subpaths()

                for (i_ent1, i_ent2), path in paths:
                    #print((i_ent1, i_ent2), path)
                    tup = (key[i_ent1], key[i_ent2])

                    # special token for empty path
                    if len(path) == 0:
                        path = ("<EMPTY>",)

                    # register tokens in path
                    for _p in path:
                        if _p != "<EMPTY>":
                            if _p not in tokens_cnt:
                                tokens_cnt[_p] = 0
                            tokens_cnt[_p] += 1
                            if tokens_cnt[_p] >= min_word_freq:
                                tokens.add(_p)

                    if tup not in observed_columns:
                        observed_columns[tup] = []
                    observed_columns[tup].append(
                        ("path", path)
                    )

                    if phase=="train":
                        if tup not in train_columns:
                            train_columns[tup] = []
                        train_columns[tup].append(
                            ("path", path)
                        )
                    #print(tup, path)

            # KB relation
            if items[phase][key]["relation"] in DATA_CONFIG["predicates"]:
                i_r = DATA_CONFIG["predicates"][items[phase][key]["relation"]]

                if phase == "train":
                    for i_ent1 in range(n_ent):
                        for i_ent2 in range(n_ent):
                            if i_ent1 == i_ent2:
                                continue

                            tup = (key[i_ent1], key[i_ent2])


                            if tup not in observed_columns:
                                observed_columns[tup] = []
                            observed_columns[tup].append(
                                ("symbol", ((i_ent1, i_ent2), i_r))
                            )

                            if tup not in train_columns:
                                train_columns[tup] = []
                            train_columns[tup].append(
                                ("symbol", ((i_ent1, i_ent2), i_r))
                            )

    tokens = ["<PAD>", "<EMPTY>", "<UNK>"] + list(tokens)
    tokens_ind = {t:i_t for i_t, t in enumerate(tokens)}

    new_observed_columns = {}
    for tup in observed_columns:
        new_observed_columns[tup] = []
        for _type, _value in observed_columns[tup]:
            if _type == "symbol":
                new_observed_columns[tup].append((_type, _value))
            elif _type == "path":
                new_observed_columns[tup].append((_type, tuple([tokens_ind[t] \
                    if t in tokens_ind else tokens_ind["<UNK>"] for t in _value])))

        new_observed_columns[tup] = list(new_observed_columns[tup])

    new_train_columns = {}
    for tup in train_columns:
        new_train_columns[tup] = []
        for _type, _value in train_columns[tup]:
            if _type == "symbol":
                new_train_columns[tup].append((_type, _value))
            elif _type == "path":
                new_train_columns[tup].append((_type, tuple([tokens_ind[t] \
                    if t in tokens_ind else tokens_ind["<UNK>"] for t in _value])))

        new_train_columns[tup] = list(new_train_columns[tup])

    return items, new_observed_columns, tokens, tokens_ind, DATA_CONFIG["predicates"],\
        DATA_CONFIG["arities"], new_train_columns

def load_data_single_text(data_json, min_word_freq=5, given_items=None):
    with open(data_json) as h:
        DATA_CONFIG = json.load(h)

    if given_items is None:
        ### Load data.
        print("Loading data...")
        items = {}
        for phase in ["train", "dev", "test"]:
            items[phase] = load_data_graph(
                DATA_CONFIG["data"].format(phase)
            )
    else:
        items = given_items

    ### Extract texts.
    observed_contexts = {}
    train_contexts = {}
    tokens = set([])
    tokens_cnt = {}
    for phase in ["train", "dev", "test"]:
        print(phase)
        for i_key, key in enumerate(items[phase]):
            if i_key % 1000 == 0:
                print("Processed {} out of {} entity tuples.".format(i_key, len(items[phase])))

            n_ent = len(key)

            # surface patterns
            for doc in items[phase][key]["docs"]:
                # Extract text of the document.
                text = [node["lemma"] for sent in doc["sentences"] for node in sent["nodes"]]
                if len(text) == 0:
                    text = ["<EMPTY>"]

                # Register tokens.
                for tok in text:
                    if tok != "<EMPTY>":
                        if tok not in tokens_cnt:
                            tokens_cnt[tok] = 0
                        tokens_cnt[tok] += 1
                        if tokens_cnt[tok] >= min_word_freq:
                            tokens.add(tok)

                # Register text to entity's list.
                for i_ent in range(n_ent):
                    ent = key[i_ent]
                    pos = tuple(doc["entities"][i_ent]["indices"])

                    if ent not in observed_contexts:
                        observed_contexts[ent] = []
                    observed_contexts[ent].append(
                        ("text",(text,pos))
                    )

                    if phase=="train":
                        if ent not in train_contexts:
                            train_contexts[ent] = []
                        train_contexts[ent].append(
                            ("text", (text, pos))
                        )

            # KB relation
            if items[phase][key]["relation"] in DATA_CONFIG["predicates"]:
                i_r = DATA_CONFIG["predicates"][items[phase][key]["relation"]]

                # Register KB relation to entity's list.
                if phase=="train":
                    for i_ent in range(n_ent):
                        ent = key[i_ent]
                        if ent not in observed_contexts:
                            observed_contexts[ent] = []
                        observed_contexts[ent].append(
                            ("symbol", (i_ent, i_r))
                        )

                        if ent not in train_contexts:
                            train_contexts[ent] = []
                        train_contexts[ent].append(
                            ("symbol", (i_ent, i_r))
                        )

    tokens = ["<PAD>", "<EMPTY>", "<UNK>"] + list(tokens)
    tokens_ind = {t:i_t for i_t, t in enumerate(tokens)}

    new_observed_contexts = {}
    for ent in observed_contexts:
        new_observed_contexts[ent] = []
        for _type, _value in observed_contexts[ent]:
            if _type == "symbol":
                new_observed_contexts[ent].append((_type, _value))
            elif _type == "text":
                _text, _pos = _value
                _new_text = tuple(
                    [tokens_ind[t] if t in tokens_ind else tokens_ind["<UNK>"]\
                        for t in _text]
                )
                new_observed_contexts[ent].append(
                    (
                        _type,
                        (_new_text, _pos)
                    )
                )

    new_train_contexts = {}
    for ent in train_contexts:
        new_train_contexts[ent] = []
        for _type, _value in train_contexts[ent]:
            if _type == "symbol":
                new_train_contexts[ent].append((_type, _value))
            elif _type == "text":
                _text, _pos = _value
                _new_text = tuple(
                    [tokens_ind[t] if t in tokens_ind else tokens_ind["<UNK>"]\
                        for t in _text]
                )
                new_train_contexts[ent].append(
                    (
                        _type,
                        (_new_text, _pos)
                    )
                )

    return items, new_observed_contexts, tokens, tokens_ind, DATA_CONFIG["predicates"],\
        DATA_CONFIG["arities"], new_train_contexts

def load_data_text(data_json, min_word_freq=5):
    with open(data_json) as h:
        DATA_CONFIG = json.load(h)

    ### Load data.
    print("Loading data...")

    items = {}
    for phase in ["train", "dev", "test"]:
        items[phase] = load_data_graph(
            DATA_CONFIG["data"].format(phase)
        )

    ### extract pairwise shortest dependency paths
    print("Loading observed columns...")
    observed_columns = {} # KB relations in train data and all surface patterns.
    tokens = set([])
    tokens_cnt = {}
    for phase in ["train", "dev", "test"]:
        print(phase)
        for i_key, key in enumerate(items[phase]):
            if i_key % 1000 == 0:
                print("Processed {} out of {} entity tuples.".format(i_key, len(items[phase])))
            #print(key)
            n_ent = len(key)

            # surface patterns
            for doc in items[phase][key]["docs"]:
                # flatten sentences
                nodes = []
                for sent in doc["sentences"]:
                    for node in sent["nodes"]:
                        token = node["label"]

                        nodes.append(token)

                        if token not in tokens_cnt:
                            tokens_cnt[token] = 0
                        tokens_cnt[token] += 1

                        if tokens_cnt[token] > min_word_freq:
                            tokens.add(token)
                nodes = tuple(nodes)

                # entity positions
                ent_pos = []
                for ent in doc["entities"]:
                    ent_pos.append(tuple(ent["indices"]))

                for i_ent1 in range(n_ent):
                    for i_ent2 in range(n_ent):
                        if i_ent1 == i_ent2:
                            continue
                        #print((i_ent1, i_ent2), path)
                        tup = (key[i_ent1], key[i_ent2])

                        if tup not in observed_columns:
                            observed_columns[tup] = []
                        observed_columns[tup].append(
                            ("sent", (nodes, (ent_pos[i_ent1], ent_pos[i_ent2])))
                        )
                        #print(tup, path)

            # KB relation
            if items[phase][key]["relation"] in DATA_CONFIG["predicates"]:
                i_r = DATA_CONFIG["predicates"][items[phase][key]["relation"]]

                if phase == "train":
                    for i_ent1 in range(n_ent):
                        for i_ent2 in range(n_ent):
                            if i_ent1 == i_ent2:
                                continue

                            tup = (key[i_ent1], key[i_ent2])

                            if tup not in observed_columns:
                                observed_columns[tup] = []
                            observed_columns[tup].append(
                                ("symbol", ((i_ent1, i_ent2), i_r))
                            )


    tokens = ["<PAD>", "<UNK>"] + list(tokens)
    tokens_ind = {t:i_t for i_t, t in enumerate(tokens)}

    new_observed_columns = {}
    for tup in observed_columns:
        new_observed_columns[tup] = []
        for _type, _value in observed_columns[tup]:
            if _type == "symbol":
                new_observed_columns[tup].append((_type, _value))
            elif _type == "sent":
                nodes, pos = _value
                nodes = tuple([tokens_ind[t] \
                    if t in tokens_ind else tokens_ind["<UNK>"] for t in nodes])
                new_observed_columns[tup].append((_type, (nodes, pos)))

        new_observed_columns[tup] = list(new_observed_columns[tup])

    return items, new_observed_columns, tokens, tokens_ind, DATA_CONFIG["predicates"], DATA_CONFIG["arities"]


def load_word_vector(dim=100):
    print("loading word vectors")
    vectors = {}
    with open(CONFIG_DIR["glove"].format(dim)) as h:
        i_line = 0
        for l in h:
            i_line += 1
            if i_line % 20000 == 0:
                print("Processed {} lines...".format(i_line))

            line = l.split()
            word = line[0]
            vect = np.array(line[1:]).astype(np.float)

            vectors[word] = vect

    return vectors

def filter_data(observed_tuples, n_rel, label_ratio=1.0, sparse_ratio=1.0):
    if label_ratio < 1.0:
        _observed_tuples = shuffle(observed_tuples)

        relation_pos_tups = defaultdict(set)
        for tup in _observed_tuples:
            p,r = tup
            if isinstance(r, int):
                relation_pos_tups[r].add(p)
        for i in range(n_rel):
            new_label_num = int(np.ceil(label_ratio * len(relation_pos_tups[i])))
            tmp_tup_lst = list(relation_pos_tups[i])
            relation_pos_tups[i] = \
                set([
                    tmp_tup_lst[_] for _ in np.random.choice(len(tmp_tup_lst),
                        size=new_label_num, replace=False)
                ])

        observed_tuples = []
        for tup in _observed_tuples:
            p,r = tup
            if isinstance(r, int):
                if p not in relation_pos_tups[r]:
                    continue
            observed_tuples.append(tup)

    if sparse_ratio < 1.0:
        set_multiple_surface_keys = set([])
        n_pattern = defaultdict(int)

        for p,r in observed_tuples:
            if not isinstance(r, int):
                n_pattern[p] += 1
                if n_pattern[p] > 1:
                    set_multiple_surface_keys.add(p)

        n_multiple_surface_keys = int(len(set_multiple_surface_keys) * sparse_ratio)
        print("Number of entity tuples with multiple surface patterns: {} -> {}".format(
            len(set_multiple_surface_keys), n_multiple_surface_keys))
        tmp_lst_key = list(set_multiple_surface_keys)
        new_set_multiple_surface_keys = np.random.choice(len(tmp_lst_key),
            size=n_multiple_surface_keys, replace=False)
        new_set_multiple_surface_keys = set([
            tmp_lst_key[_] for _ in new_set_multiple_surface_keys
        ])

        set_filter_keys = set_multiple_surface_keys - new_set_multiple_surface_keys
        set_filtered_keys = set([])

        new_observed_tuples = []
        for tup in observed_tuples:
            p,r = tup
            if not isinstance(r,int):
                if (p in set_filter_keys) and (p in set_filtered_keys):
                    continue
                set_filtered_keys.add(p)
            new_observed_tuples.append(tup)
        observed_tuples = new_observed_tuples

    return observed_tuples

if __name__=="__main__":
    pass
