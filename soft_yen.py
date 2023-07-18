import pandas as pd
import numpy as np
from igraph import Graph

def soft_yen_refactored(g: Graph, source: int, target: int, g_reference: Graph = None, mode: str = 'subpath',
             score_mode: str = 'prod', n_cycles: int = 10, penalty_coef: float = 0.95,
             label: str = 'label', return_df: bool = False):
    """
    This algorithm is a variant of Yen's algorithm. It searches for the shortest path (reconstructed signaling path, RSP)
    from the target gene to genes associated with diseases or biological processes. It can be drug-targets, GWAS or top
    differentially expressed genes. Our algorithm uses a weighted undirected PPI graph.
    Edge weights have to be between 0 and 1. The higher the value, the higher its confidence level.
    For ranking RSPs, we use the cumulative score of weights of edges coming to it.

    :param g: igraph graph instance
    :param source: index of the source vertex in the graph g
    :param target: index of the target vertex in the graph g
    :param g_reference: reference graph with the same topology as g, but possibly with different edge weights.
                        It is used to calculate the final cumulative score for a generated hypothesis.
    :param mode: string, either 'subpath', 'best' or 'worst' - how to update weights on the edges already visited while
                 generating previous best RSPs. If 'subpath' is used (default) - all visited edges will be updated.
                 If 'best' is used - only visited edge with the lowest weight will be updated. If 'worst' is used - only edge
                 with the worst (highest) weight in the previous RSP will be updated.
    :param score_mode: string, either 'prod' or 'sum' - how to calculate the cumulative score of the RSP.
    :param n_cycles: number of raw hypotheses generated. Since the same pathway may appear twice (if the weight update
                     on its edges is not sufficiently high) - the actual final number of generated hypothesis may be lower.
    :param penalty_coef: a float, coefficient on which each edge in the previously selected best hypotheses will be
                         divided to increase edge weights on it and thus penalize repetitive selection of the same subpath.
    :param label: string, name of the graph vertex attribute in which gene names are stored. default = 'label',
    :param return_df: bool, if True - returns RSPs and their respective scores in the form of a data frame
    :return:
        paths: python list containing RSPs
        scores: python list containing scores for each RSP
    """
    assert mode in ('subpath', 'best', 'worst'), "mode should be one of ('subpath', 'best', 'worst')"
    assert score_mode in ('sum', 'prod'), "mode should be one of ('sum', 'prod')"
    assert 0 < penalty_coef < 1, 'penalty coefficient must be above zero and below one'

    if not g_reference:
        g_reference = g.copy()

    if 'weight' not in g.es()[0].attributes():
        g.es()['weight'] = 1

    if 'weight' not in g_reference.es()[0].attributes():
        g_reference.es()['weight'] = 1

    # Reversing edges weight for correctly working algorithm
    g.es()['weight'] = 1 / np.array(g.es()['weight'])

    paths = []
    scores = []
    min_l = 1000000
    for cycle in range(n_cycles):
        # get nodes list for RSP
        path = g.get_shortest_paths(source, to=target, weights='weight')[0]
        if len(path) > min_l:
            break
        else:
            min_l = len(path)
        # get edges list for RSP
        epath = g.get_shortest_paths(source, to=target, weights='weight', output='epath')[0]

        # increase the cost of going through the links through which the RSP was laid
        new_weights = np.array(g.es[epath]['weight']) / penalty_coef

        if mode == 'subpath':
            g.es[epath]['weight'] = new_weights
        elif mode == 'best':
            idx = np.argmin(new_weights)
            g.es[idx]['weight'] = new_weights[idx]
        else:
            idx = np.argmax(new_weights)
            g.es[idx]['weight'] = new_weights[idx]

        # compute cumulative score for RSP
        if score_mode == 'prod':
            score = np.prod(g_reference.es[epath]['weight'])
            score = np.log(score)
        else:
            score = np.sum(g_reference.es[epath]['weight'])

        paths.append(path)
        scores.append(score)

    # print(paths)
    _, paths_ids = np.unique([set(k) for k in paths], return_index=True)
    # print([paths[i] for i in paths_ids])
    # paths = np.array([paths[i] for i in paths_ids])
    paths = [paths[i] for i in paths_ids]
    scores = [scores[i] for i in paths_ids]
    scores = np.array(scores)
    ids = np.argsort(scores)
    paths, scores = [paths[idx] for idx in ids], scores[ids]

    if return_df:
        # transform dict of RSP to dataframe
        paths = [g.vs(p)[label] for p in paths]
        paths = [';'.join(p) for p in paths]
        hyp = {p:s for p,s in zip(paths, scores)}
        df = pd.DataFrame.from_dict(hyp, orient='index')
        df = df.reset_index()
        df.columns = ['path', 'path_score']
        df = df.sort_values(by="path_score",ascending=False)
        return df

    else:
        return paths, scores