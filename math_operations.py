import networkx as nx
import numpy as np
import logger
import sklearn
import tqdm
            

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
    M = nx.to_scipy_sparse_matrix(G, dtype=float)
    if normalization == "auto":
        normalization = "col" if G.is_directed() else "symmetric"
    if normalization == "col":
        M = sklearn.preprocessing.normalize(M, "l1", axis=1, copy=False)
    elif normalization == "symmetric":
        M = sklearn.preprocessing.normalize(M, "l2", axis=0, copy=False)
        M = sklearn.preprocessing.normalize(M, "l2", axis=1, copy=False)
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
    #return {nodei: {nodej: result[i,j] for j, nodej in enumerate(G.nodes())} for i, nodei in enumerate(G.nodes())}



def transpose(x):
    return [[(x[j][i]) for j in range(len(x))] for i in range(len(x[0]))] 

                                