import numpy as np
import matplotlib.pyplot as plt
import copy
import training
import parameters
import logger
import time
import math_operations as math_ops
import math


class Rectangle:
    def __init__(self, G, axes=[], counter=0, predictor=None, middle_rec=False, inherited_rec=[]):
        self.G = G
        self.axes = axes
        self.dims = len(axes)
        self.dim_len = [i[1]-i[0] for i in axes]
        self.size = np.prod(self.dim_len)
        self.centre = [(x[0]+(x[1]-x[0])/2) if (i!=0) else math_ops.apply_log((x[0]+(x[1]-x[0])/2), len(math_ops.apply_filter(G, [1], "symmetric", False))) for i,x in enumerate(axes)]
        # the last dimension is not a filter parameter, its the embedding dimension. It follows logarithmic values
        self.diag = np.sqrt(sum([(self.centre[i]-self.axes[i][0])**2 for i in range(self.dims)]))
        self.pars = parameters.parameters(G, False, False, [], [], True, [self.centre[1:]], [int(self.centre[0])], 1, math.inf)
        if middle_rec:
            self.auc = inherited_rec.auc
            self.sihl = inherited_rec.sihl
            self.loss = inherited_rec.loss
            self.configs = inherited_rec.configs
            self.size_list = inherited_rec.size_list
            self.diag_list = inherited_rec.diag_list
        else:
            results, configs, predictor = training.train(G, self.pars, predictor)
            self.auc = results[0]
            self.sihl = results[1]
            self.loss = results[2]
            self.configs = configs
            self.size_list = [np.nan for x in range(1 + 2*counter)]
            self.diag_list = [np.nan for x in range(1 + 2*counter)]
        
              
def find_optimal_rectangles(rectangles, epsilon):    
    potentially_optimal_indexes, metrics = [], []
    for rectangle in rectangles:
        metrics.append(rectangle.loss)
    min_loss = min(metrics)
    
    if len(rectangles) > 2:
        for r, rectangle in enumerate(rectangles):
            if criterion(rectangle, rectangles, epsilon, min_loss):
                potentially_optimal_indexes.append(r)
    else: 
        for r in range(len(rectangles)): potentially_optimal_indexes.append(r)
        
    return potentially_optimal_indexes


def criterion(rec, rectangles, epsilon, min_f):
    flag = True
    rec_bigger, rec_equal, rec_smaller = [], [], []
    for r in rectangles:
        if r.diag > rec.diag:
            rec_bigger.append(r)
        elif r.diag < rec.diag:
            rec_smaller.append(r)
        elif r.diag == rec.diag and not r == rec:
            rec_equal.append(r)

    for r in rec_equal:
        if rec.loss > r.loss: flag = False
        if flag == False: return flag
        
    a = math.inf
    b = 0   
    for r in rec_smaller:
        if (rec.loss - r.loss)/(rec.diag - r.diag) > b: b = (rec.loss - r.loss)/(rec.diag-r.diag)
    for r in rec_bigger:
        if (r.loss - rec.loss)/(r.diag- rec.diag) < a: a = (r.loss - rec.loss)/(r.diag- rec.diag)
        
    if a <= b: flag = False
    if epsilon > (min_f - rec.loss)/abs(min_f) + rec.diag/abs(min_f)*a: flag = False
    
    return flag


def trisect(indexes, rectangles, counter, stop_condition_perc, trisection_lim, predictor):
    logger.log("Starting", len(indexes), "trisections for this iteration of DIRECT algorithm.") if len(indexes)>=2 else logger.log("Starting", len(indexes), "trisection for this iteration of DIRECT algorithm.")
    rectangles_to_trisect = [rectangles[i] for i in indexes]
    stop, stop_mess = False, ""
    for rectangle in rectangles_to_trisect:
        start = time.time()
        logger.log("\n\nPERFORMING TRISECTION NUMBER", counter+1)
        index_max = np.argmax(rectangle.dim_len)
        axes1, axes2, axes3 = copy.deepcopy(rectangle.axes), copy.deepcopy(rectangle.axes), copy.deepcopy(rectangle.axes)
        axes1[index_max][1] = axes1[index_max][0]+(axes1[index_max][1]-axes1[index_max][0])/3
        axes2[index_max][0], axes2[index_max][1] = axes2[index_max][0]+(axes2[index_max][1]-axes2[index_max][0])/3, axes2[index_max][0]+2*(axes2[index_max][1]-axes2[index_max][0])/3
        axes3[index_max][0] = axes3[index_max][0]+2*(axes3[index_max][1]-axes3[index_max][0])/3
        rectangle1, rectangle2, rectangle3  = Rectangle(rectangle.G, axes1, counter, predictor), Rectangle(rectangle.G, axes2, counter, predictor, True, rectangle), Rectangle(rectangle.G, axes3, counter, predictor)
        rectangles[rectangles.index(rectangle)] = rectangle2 # inserting the middle rectangle in the place of the old, so to correspond with the other stats (training stats)
        rectangles.extend([rectangle1, rectangle3])
        counter += 1
        rectangles = renew_size_list(rectangles, counter)
        end = time.time()
        logger.log("\nThe time it took for this trisection was", end-start, "secs.")
        stop_mess = stop_mess+str(((abs(rectangle2.loss-rectangle1.loss))/rectangle2.loss)*100)+" ~ "+str(stop_condition_perc)+" and "+str(((abs(rectangle2.loss-rectangle3.loss))/rectangle2.loss)*100)+" ~ "+str(stop_condition_perc) + "\n"
        if ((abs(rectangle2.loss-rectangle1.loss))/rectangle2.loss < stop_condition_perc/100 or (abs(rectangle2.loss-rectangle3.loss))/rectangle2.loss < stop_condition_perc/100): stop = True
    if (counter >= trisection_lim): stop = True
    if stop: logger.log(stop_mess)
    return rectangles, counter, stop, predictor


def renew_size_list(rectangles, counter):
    for rec in rectangles:
        for i in range(1 + 2*counter - len(rec.size_list)):
            rec.size_list.extend([rec.size])
            rec.diag_list.extend([rec.diag])
    return rectangles
        
        
def draw_rectangles(rectangles):
    """    draws all the parameters compared to the 1st filter parameter. for ex. filter par 1-embed dim, filter par 1- filter par 2, filter par 1- filter par 3, filter par 1 - filter par 4"""
    centres, array = [[] for i in range(len(rectangles))], [[[] for i in range(rectangles[0].dims)] for j in range(len(rectangles))]
    for r,rectangle in enumerate(rectangles):       
        for i in range(rectangle.dims):
            if i == 1:
                array[r][i].extend([rectangle.axes[i][0], rectangle.axes[i][0], rectangle.axes[i][1], rectangle.axes[i][1], rectangle.axes[i][0]])
                centres[r].extend([rectangle.centre[i]])
            else:
                array[r][i].extend([rectangle.axes[i][0], rectangle.axes[i][1], rectangle.axes[i][1], rectangle.axes[i][0], rectangle.axes[i][0]])
                centres[r].extend([rectangle.centre[i]])

        fig2 = plt.figure(1, figsize=((10*(rectangle.dims)), 10))
        for i in range(0, rectangle.dims-1):      
            plt.subplot(1, int(rectangle.dims-1), i+1)
            plt.xlabel("filter parameter: 1")
            if i == 0:
                plt.ylabel("embedding dimension")
                plt.yscale('log', basey=2)
                plt.plot(array[r][1], [math_ops.apply_log(x, len(math_ops.apply_filter(rectangle.G, [1], "symmetric", False))) for x in array[r][i]])
                plt.scatter(centres[r][1], centres[r][i])
            else:
                plt.ylabel("filter parameter:%s"%(i+1)) 
                plt.plot(array[r][1], array[r][i+1])
                plt.scatter(centres[r][1], centres[r][i+1])
                
    plt.show(block=False)
        
    return fig2


def sort_rectangles(rectangles_original, metric):        
    metrics, configs, rectangles = [], [], copy.deepcopy(rectangles_original)
    for rec in rectangles:
        if metric == "loss":
            metrics.append(rec.loss)
        elif metric == "auc":
            metrics.append(rec.auc)
        elif metric == "sihlouette":
            metrics.append(rec.sihl)
        configs.append(rec.configs)


    # simple selection sort       
    for i in range(len(metrics)):
        minimum = i        
        for j in range(i + 1, len(metrics)):
            if metrics[j] < metrics[minimum]:
                minimum = j
        metrics[minimum], metrics[i], rectangles[minimum], rectangles[i], configs[minimum], configs[i] = metrics[i], metrics[minimum], rectangles[i], rectangles[minimum], configs[i], configs[minimum]
        
    sorted_results = []
    for m,c,r in zip(metrics, configs, rectangles):
        sorted_results.append((m,c,r))
               
    return sorted_results
    
   