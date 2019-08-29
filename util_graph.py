# coding: UTF-8

from functools import reduce

import networkx as nx

try:
    import pygraphviz as pgv
except Exception:
    print("No graphviz")
    pass

def create_document_graph(document_item):
    # Create Document Graph
    graph = nx.DiGraph()

    # Add nodes
    nodes = reduce(lambda x,y: x+y, [sent["nodes"] for sent in document_item["sentences"]])
    for i_node in range(len(nodes)):
        graph.add_node(i_node, label=nodes[i_node]["lemma"])

    # Add edge
    for i_node, node in enumerate(nodes):
        for arc in node["arcs"]:
            link = arc["label"]
            if link[:8] == "adjsent:":
                # Connection via coreference is better than simple adjacency.
                weight = 10.0
            elif link[:6] == "coref:":
                weight = 1.0
            elif (link[:7] == "deparc:") or (link[:7] == "depinv:"):
                weight = 1.0
            else:
                continue
            from_ind = i_node
            to_ind = arc["toIndex"]

            # Skip link to ROOT
            if min(from_ind, to_ind) < 0:
                continue

            graph.add_edge(from_ind, to_ind, label=link, weight=weight)

    return graph

def draw_document_graph(document_item, output="tmp.png"):
    graph = create_document_graph(document_item)

    for edge in graph.edges:
        graph[edge[0]][edge[1]]["fontsize"] = 10

    g = nx.nx_agraph.to_agraph(graph)
    g.draw(output, prog="fdp")

class PatternPairwiseShortestPath(object):
    def __init__(self, doc_graph, n_entity, min_paths):
        u"""
        min_path: (from_ent_ind, to_ent_ind) -> path
        path is list of index of node in the document graph
        """
        def get_link_name(i_from, i_to):
            link = doc_graph.edges[i_from, i_to]["label"]
            flip = False
            arrow = "<-" if flip else "->"
            link = "{}{}{}".format(arrow, link, arrow)

            return link

        contained_links = [] # list of (from_ind, to_ind, link_label)
        new_paths_dict = {}
        node_ind_lst = []
        node_lemma_lst = []
        new_paths = []
        for i_ent1 in range(n_entity):
            for i_ent2 in range(n_entity):
                if i_ent2 == i_ent1:
                    continue

                path = min_paths[i_ent1, i_ent2]

                for node_ind in path[1:-1]:
                    if node_ind not in node_ind_lst:
                        node_ind_lst.append(node_ind)
                        node_lemma_lst.append(doc_graph.nodes[node_ind]["label"])

                new_path = []
                for i_pos in range(len(path)-1):
                    if i_pos != 0:
                        new_path.append(node_ind_lst.index(path[i_pos]))

                    link = get_link_name(path[i_pos], path[i_pos+1])
                    new_path.append(link)

                    i_from, i_to = path[i_pos], path[i_pos+1]
                    contained_links.append(
                        (i_from, i_to, doc_graph.edges[i_from, i_to]["label"])
                    )
                new_paths.append(tuple(new_path))
                new_paths_dict[(i_ent1, i_ent2)] = tuple(new_path)
        self.node_lemma = tuple(node_lemma_lst)
        self.new_paths = tuple(new_paths)
        self.contained_links = set(contained_links)
        self.new_paths_dict = new_paths_dict

    def get_subpaths(self):
        paths = []
        for _k, _p in self.new_paths_dict.items():
            p = []
            for tok in _p:
                if isinstance(tok, int):
                    tok = self.node_lemma[tok]
                p.append(tok)
            paths.append((_k, tuple(p)))
        return paths

    def __eq__(self, other):
        if not isinstance(other, PatternPairwiseShortestPath):
            return False
        return (self.node_lemma == other.node_lemma) and (self.new_paths == other.new_paths)

    def __hash__(self):
        return hash((self.node_lemma, self.new_paths))

    def __str__(self):
        return str((self.node_lemma, self.new_paths))


def extract_pairwise_shortest_paths(document_item):
    if min([len(ent["indices"]) for ent in document_item["entities"]]) < 1:
        print("There is an entity with no indices.")
        return None

    doc_graph = create_document_graph(document_item)

    # Find pairwise shortest paths between each entity pairs.
    min_paths = {}
    for i_ent1, entity1 in enumerate(document_item["entities"]):
        for i_ent2, entity2 in enumerate(document_item["entities"]):
            if i_ent2 == i_ent1:
                continue

            # Find shortest path
            min_length = None
            min_path = None
            for from_ind in entity1["indices"]:
                for to_ind in entity2["indices"]:
                    try:
                        path = nx.dijkstra_path(doc_graph, from_ind, to_ind, "weight")
                    except nx.exception.NetworkXNoPath:
                        continue
                    path_len = len(path) - 1
                    if (min_length is None) or (path_len < min_length):
                        min_length = path_len
                        min_path = path

            if min_path is None:
                draw_document_graph(document_item, "error.png")
                draw_document_graph(document_item, "error2.png")
                print("Document graph is dumped in error.png")
                print("Entities are: {}".format(document_item["entities"]))
                raise Exception("No shortest path between entity {} and {}".format(i_ent1, i_ent2))
            min_paths[i_ent1, i_ent2] = min_path

    return PatternPairwiseShortestPath(doc_graph, len(document_item["entities"]),
            min_paths)

if __name__=="__main__":
    from util import load_data

    items, indmap, _observed_tuples, arities = load_data("wikismall.data.json", tuple_type="ent_index")

    for key in items["train"]:
        sample_doc = items["train"][key]["docs"][0]

        tokens = []
        for sent in sample_doc["sentences"]:
            for node in sent["nodes"]:
                tokens.append(node["label"])
        raw_sent = " ".join(tokens)

        ents = [ent for ent in sample_doc["entities"]]
        print(ents)

        print(raw_sent)

        pair = extract_pairwise_shortest_paths(sample_doc)
        for sp in pair.get_subpaths():
            print(sp)

        print("==============================")
        input()
