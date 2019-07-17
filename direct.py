import numpy as np
import matplotlib.pyplot as plt
import copy
import training
import parameters
import logger
import time
import math_operations as mo


class Rectangle:
    def __init__(self, G, axes=[], predictor=None, middle_rec=False, inherited_data=[]):
        self.G = G
        self.axes = axes
        self.dims = len(axes)
        self.dim_len = [i[1]-i[0] for i in axes]
        self.size = np.prod(self.dim_len)
        self.centre = [(x[0]+(x[1]-x[0])/2) if (i!=len(axes)-1) else mo.apply_log((x[0]+(x[1]-x[0])/2)) for i,x in enumerate(axes)]
        # the last dimension is not a filter parameter, its the embedding dimension. It follows logarithmic values
        self.pars = parameters.parameters(G, False, False, [], [], True, [self.centre[:-1]], [int(self.centre[-1])], 1, 5000)
        if middle_rec:
            self.auc = inherited_data[0]
            self.sihl = inherited_data[1]
            self.loss = inherited_data[2]
            self.configs = inherited_data[3]
        else:
            results, configs, predictor = training.train(G, self.pars, predictor)
            self.auc = results[0]
            self.sihl = results[1]
            self.loss = results[2]
            self.configs = configs
        
              
def find_optimal_rectangles(rectangles, k):
    
    potentially_optimal_indexes, metrics, sizes = [], [], []
    for rectangle in rectangles:
        metrics.append(rectangle.loss)
        sizes.append(rectangle.size)
    min_metrics = min(metrics)
    min_metrics_size = sizes[metrics.index(min_metrics)]
    
    if len(rectangles) > 2:
        for r, rectangle in enumerate(rectangles):
            if (rectangle.size - metrics[r]*k >= min_metrics_size - min_metrics*k) and (True in [l>0.1 for l in rectangle.dim_len[:-1]]):
                potentially_optimal_indexes.append(r)
    else: 
        for r in range(len(rectangles)): potentially_optimal_indexes.append(r)
        
    return potentially_optimal_indexes


def trisect(indexes, rectangles, counter, predictor):
    
    rectangles_to_trisect = [rectangles[i] for i in indexes]
    for rectangle in rectangles_to_trisect:
        start = time.time()
        logger.log("PERFORMING TRISECTION NUMBER", counter+1)
        counter += 1
        index_max = np.argmax(rectangle.dim_len)
        axes1, axes2, axes3 = copy.deepcopy(rectangle.axes), copy.deepcopy(rectangle.axes), copy.deepcopy(rectangle.axes)
        axes1[index_max][1] = axes1[index_max][0]+(axes1[index_max][1]-axes1[index_max][0])/3
        axes2[index_max][0], axes2[index_max][1] = axes2[index_max][0]+(axes2[index_max][1]-axes2[index_max][0])/3, axes2[index_max][0]+2*(axes2[index_max][1]-axes2[index_max][0])/3
        axes3[index_max][0] = axes3[index_max][0]+2*(axes3[index_max][1]-axes3[index_max][0])/3
        rectangle1, rectangle2, rectangle3  = Rectangle(rectangle.G, axes1, predictor), Rectangle(rectangle.G, axes2, predictor, True, [rectangle.auc, rectangle.sihl, rectangle.loss, rectangle.configs]), Rectangle(rectangle.G, axes3, predictor)
        rectangles[rectangles.index(rectangle)] = rectangle2 # inserting the middle rectangle in the place of the old, so to correspond with the other stats (training stats)
        rectangles.extend([rectangle1, rectangle3])
        end = time.time()
        logger.log("\nThe time it took for this trisection was", end-start, "secs.")
    return rectangles, counter, predictor
     
        
def draw_rectangles_1(rectangles):
    """
    draws all the parameters compared to the 1st. for ex. 1-2 1-3 1-4
    """
    centres, array = [[] for i in range(len(rectangles))], [[[] for i in range(rectangles[0].dims)] for j in range(len(rectangles))]
    for r,rectangle in enumerate(rectangles):
        
        for i in range(rectangle.dims):
            if i == 0:
                array[r][0].extend([rectangle.axes[0][0], rectangle.axes[0][0], rectangle.axes[0][1], rectangle.axes[0][1], rectangle.axes[0][0]])
                centres[r].extend([rectangle.centre[0]])
            else:
                array[r][i].extend([rectangle.axes[i][0], rectangle.axes[i][1], rectangle.axes[i][1], rectangle.axes[i][0], rectangle.axes[i][0]])
                centres[r].extend([rectangle.centre[i]])

        fig2 = plt.figure(1, figsize=((2*rectangle.dims)*(rectangle.dims-1), 2*rectangle.dims))
        for i in range(1, rectangle.dims):      
            plt.subplot(1, int(rectangle.dims-1), i)
            plt.xlabel("filter parameter: 1")
            if i == int(rectangle.dims-1):
                plt.ylabel("embedding dimension")
                plt.yscale('log', basey=2)
                plt.plot(array[r][0], [mo.apply_log(x) for x in array[r][i]])
                plt.scatter(centres[r][0], centres[r][i])
            else:
                plt.ylabel("filter parameter:%s"%i) 
                plt.plot(array[r][0], array[r][i])
                plt.scatter(centres[r][0], centres[r][i])
                
    plt.show(block=False)
        
    return fig2


def draw_rectangles_2(rectangles):
    """
    draws every 2nd variable with the next. for ex. 1-2 3-4 etc
    """
    centres, array = [[] for i in range(len(rectangles))], [[[] for i in range(rectangles[0].dims)] for j in range(len(rectangles))]
    for r,rectangle in enumerate(rectangles):
        
        for i in range(0,rectangle.dims,2):
            array[r][i].extend([rectangle.axes[i][0], rectangle.axes[i][0], rectangle.axes[i][1], rectangle.axes[i][1], rectangle.axes[i][0]])
            array[r][i+1].extend([rectangle.axes[i+1][0], rectangle.axes[i+1][1], rectangle.axes[i+1][1], rectangle.axes[i+1][0], rectangle.axes[i+1][0]])
            centres[r].extend([rectangle.centre[i], rectangle.centre[i+1]])

        fig2 = plt.figure(1, figsize=((2*rectangle.dims)*(rectangle.dims/2), 2*rectangle.dims))
        for i in range(0,rectangle.dims,2):
            plt.xlabel("dimension: %s"%(i+1))
            plt.ylabel("dimension: %s"%(i+2))            
            plt.subplot(1, int(rectangle.dims/2), (i+2)/2)
            plt.plot(array[r][i], array[r][i+1])
            plt.scatter(centres[r][i], centres[r][i+1])
    plt.show(block=False)
        
    return fig2


def sort_rectangles(rectangles, metric):
        
    metrics = []
    for rec in rectangles:
        if metric == "loss":
            metrics.append(rec.loss)
        elif metric == "auc":
            metrics.append(rec.auc)
        elif metric == "sihlouette":
            metrics.append(rec.sihl)

    # simple selection sort       
    for i in range(len(metrics)):
        minimum = i        
        for j in range(i + 1, len(metrics)):
            if metrics[j] < metrics[minimum]:
                minimum = j
        metrics[minimum], metrics[i], rectangles[minimum], rectangles[i] = metrics[i], metrics[minimum], rectangles[i], rectangles[minimum]
        
    sorted_filters = []    
    for i in rectangles:
        sorted_filters.append(i)
        
    return rectangles, metrics, sorted_filters
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

        