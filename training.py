import numpy as np
import topic_analysis as topic
import time
import math_operations as math_ops
import math
import sklearn.metrics
import logger


def train(G, pars, predictor=None):    
    total_experiments = len(pars.filters)*len(pars.emb_dims)
    logger.log("Starting ", total_experiments, "experiments.\n") if total_experiments!=1 else logger.log("Starting ", total_experiments, "experiment.\n")
    logger.log("Embedding dimension is", pars.emb_dims[0])
    
    results, configurations = np.zeros((3, len(pars.filters)*len(pars.emb_dims)*pars.iterations)), []
    if pars.iterations <= 0:
        raise Exception("Iterations must be a positive number")
    total_counter = 0
    
    start_out = time.time()
    for filt in pars.filters:
        for dim in pars.emb_dims:
            x_train = np.identity(len(G))
            y_train = math_ops.apply_filter(G, filt)
            node2id = {node: i for i,node in enumerate(G.nodes())}
            id2node = {node2id[node]: node for node in G.nodes()}
            function_names = [node for i, node in enumerate(G.nodes())]
            for i in range(pars.iterations): 
                logger.log("\n\nStarting iteration number", i+1, "out of", pars.iterations)
                start_in = time.time()
                vectors, loss, predictor, complete_training_bool = topic.encode(y_train, x_train, dim, pars.epochs, "Entropy", predictor)
                end_in = time.time()
                logger.log("Time it took to train:", end_in-start_in, "seconds")              
                results[0][total_counter], results[1][total_counter], results[2][total_counter] =  complete_training_bool.return_metrics(loss, vectors, id2node, function_names, G)
                configurations.append((filt, dim))
                total_counter += 1            
    end_out = time.time()  

    logger.log("The time it took to complete all experiments was", end_out - start_out, "secs, or", (end_out - start_out)/3600, "hours.\n\n")  
    
    return refine_results(results), configurations, predictor


def refine_results(results):
    """ 'erases' all dimensions of length 1 from results array """
    dims, flag = [], False
    for dim in results.shape:
        if dim != 1:
            dims.append(dim)
            flag = True
    if flag == False: dims = [1]
    return np.reshape(results, tuple(dims))




    