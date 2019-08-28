import networkx as nx
import numpy as np
import logger
            

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


def apply_filter(G, created_filter, normalize="symmetric", print_bool=True):
    adjacency_matrix = nx.to_numpy_matrix(G)
    if normalize=="none":
        pass
    elif normalize=="symmetric":
        diags = adjacency_matrix.sum(axis=0)
        diags[np.nonzero(diags)] = np.power(diags[np.nonzero(diags)],-0.5)
        D = np.diagflat(diags)
        adjacency_matrix = np.matmul(np.matmul(D.T, adjacency_matrix), D)
    else:
        raise Exception("Invalid normalization")
    power = adjacency_matrix.copy()
    result = 0
    spectrum = []
    for parameter in created_filter:
        spectrum.append(np.sum(abs(power)))
        result += power*parameter
        power = np.matmul(power, adjacency_matrix)
    result /= sum(created_filter)
    if print_bool:
        logger.log("Filter", created_filter)
        logger.log("Spectrum", spectrum)
    return {nodei: {nodej: result.item((i,j)) for j, nodej in enumerate(G.nodes())} for i, nodei in enumerate(G.nodes())}


def apply_log(x, nodes_sum):
    return int(4+2**(x*10))

def transpose(x):
    return [[(x[j][i]) for j in range(len(x))] for i in range(len(x[0]))] 

                                