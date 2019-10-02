import networkx as nx
import numpy as np
import logger
import sklearn
import tqdm
import warnings
import scipy

            

def create_filter(category="uniform", length=5, custom_filter_pars = []):
    if category=="uniform":
        filtering_algorithm = lambda x: 1
    elif category=="last":
        filtering_algorithm = lambda x: 1 if x==length-1 else 0
    elif category=="exponential":
        filtering_algorithm = lambda x: np.exp(-x*4/length)
    elif category=="heat":
        filtering_algorithm = lambda x: 0.5**x/np.math.factorial(x)
    elif category=="linear":
        filtering_algorithm = lambda x : (-x+length)/length
    elif category=="polynomial":
        filtering_algorithm = lambda x : 0.025*x**3 - 0.28*x**2 + 0.75*x + 0.3
    elif category=="custom":
        return custom_filter_pars
    else:
        raise Exception("Invalid filter category")
    return [filtering_algorithm(x) for x in range(length)]


def apply_filter(G, created_filter, normalization="symmetric", print_bool=True):        
    normalization = normalization.lower()
    M = nx.to_scipy_sparse_matrix(G, weight="weight", dtype=float, format="coo")
    if normalization == "auto":
        normalization = "col" if G.is_directed() else "symmetric"
    if normalization == "col":
        M = sklearn.preprocessing.normalize(M, "l1", axis=1, copy=False)
    elif normalization == "symmetric":
        S = scipy.array(np.sqrt(M.sum(axis=1))).flatten()
        S[S != 0] = 1.0 / S[S != 0]
        Qleft = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
        S = scipy.array(np.sqrt(M.sum(axis=0))).flatten()
        S[S != 0] = 1.0 / S[S != 0]
        Qright = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
        M = Qleft * M * Qright
    elif normalization != "none":
        raise Exception("Supported normalizations: none, col, symmetric, auto")
        
    power = M.copy()
    result = 0
    spectrum = []
    for parameter in tqdm.tqdm(created_filter, desc="Applying filter"):
        spectrum.append(np.sum(np.abs(power)))
        result += power*parameter
        power = power * M
    result /= sum(created_filter)
    result = result.todense()
    if print_bool:
        logger.log("Spectrum", spectrum)
    return result


def transpose(x):
    return [[(x[j][i]) for j in range(len(x))] for i in range(len(x[0]))] 


class LinkAUC:
    def __init__(self, G, nodes=None):
        self.G = G
        self.nodes = list(G) if nodes is None else list(set(list(nodes)))
        if self.G.is_directed():
            warnings.warn("LinkAUC is designed for undirected graphs", stacklevel=2)


    def evaluate(self, ranks, max_negative_samples=2000):
        negative_candidates = list(self.G)
        if len(negative_candidates) > max_negative_samples:
            negative_candidates = np.random.choice(negative_candidates, max_negative_samples)
        real = list()
        predicted = list()
        for node in tqdm.tqdm(self.nodes, desc="LinkAUC"):
            neighbors = self.G._adj[node]
            for positive in neighbors:
                real.append(1)
                predicted.append(np.dot(ranks[node], ranks[positive]))
            for negative in negative_candidates:
                if negative != node and negative not in neighbors:
                    real.append(0)
                    predicted.append(np.dot(ranks[node], ranks[negative]))
        fpr, tpr, _ = sklearn.metrics.roc_curve(real, predicted)
        return sklearn.metrics.auc(fpr, tpr)

                                