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
    def __init__(self, G, axes=[], counter=0, direct_iteration=0, train_iterations=1, parent_index=np.nan, predictor=None, middle_rec=False, inherited_rec=[]):
        self.G = G
        self.axes = axes
        self.dims = len(axes)
        self.dim_len = [i[1]-i[0] for i in axes]
        self.size = np.prod(self.dim_len)
        self.centre = [(x[0]+(x[1]-x[0])/2) for i,x in enumerate(axes)]
        # the last dimension is not a filter parameter, its the embedding dimension. It follows logarithmic values
        self.diag = np.sqrt(sum([(self.centre[i]-self.axes[i][0])**2 for i in range(self.dims)]))
        self.iterations = train_iterations
        self.pars = parameters.parameters(G, False, False, [], [], True, [self.centre], [32], self.iterations, 500)
        if middle_rec:
            self.auc = inherited_rec.auc
            self.sihl = inherited_rec.sihl
            self.loss = inherited_rec.loss
            self.aucs = inherited_rec.aucs
            self.sihls = inherited_rec.sihls
            self.losses = inherited_rec.losses
            self.configs = inherited_rec.configs
            self.rec_size_list = inherited_rec.rec_size_list
            self.rec_diag_list = inherited_rec.rec_diag_list
            self.optimal = inherited_rec.optimal
            self.created_in_iteration = inherited_rec.created_in_iteration
            self.parent_index = inherited_rec.parent_index
            #self.train_size_list = inherited_rec.train_size_list
            #self.train_diag_list = inherited_rec.train_diag_list
        else:
            results, configs, predictor = training.train(G, self.pars, predictor)
            self.aucs = results[1] if self.iterations>1 else [results[1]]
            self.sihls = results[0] if self.iterations>1 else [results[0]]
            self.losses = results[2] if self.iterations>1 else [results[2]]
            self.auc = (sum([x for x in results[1] if x!= -math.inf])/len([x for x in results[1] if x!= -math.inf]) if len([x for x in results[1] if x!= -math.inf])!=0 else -math.inf) if self.iterations>1 else results[1]
            self.sihl = (sum([x for x in results[0] if x!= -math.inf])/len([x for x in results[0] if x!= -math.inf]) if len([x for x in results[0] if x!= -math.inf])!=0 else -math.inf) if self.iterations>1 else results[0]
            self.loss = sum([x for x in results[2]])/len([x for x in results[2]]) if self.iterations>1 else results[2]
            self.configs = configs
            self.rec_size_list = [np.nan for x in range(1 + 2*counter)]
            self.rec_diag_list = [np.nan for x in range(1 + 2*counter)]
            self.optimal = [np.nan for x in range(1 + 2*counter)]
            self.created_in_iteration = direct_iteration
            self.parent_index = parent_index
            #self.train_size_list = [np.nan for x in range(1*self.iterations + 2*counter*self.iterations)]
            #self.train_diag_list = [np.nan for x in range(1*self.iterations + 2*counter*self.iterations)]

        
              
def find_optimal_rectangles(rectangles, epsilon):    
    potentially_optimal_indexes, metrics = [], []
    for rectangle in rectangles:
        metrics.append(rectangle.loss)
    min_loss = min(metrics)
    
    if len(rectangles) > 2:
        for r, rectangle in enumerate(rectangles):
            if criterion(rectangle, rectangles, epsilon, min_loss): potentially_optimal_indexes.append(r)
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


def trisect(indexes, rectangles, counter, direct_iteration, trisection_lim, train_iterations, predictor):
    logger.log("Starting", len(indexes), "trisections for this iteration of DIRECT algorithm.") if len(indexes)>=2 else logger.log("Starting", len(indexes), "trisection for this iteration of DIRECT algorithm.")
    rectangles_to_trisect = [rectangles[i] for i in indexes]
    for rectangle in rectangles_to_trisect:
        start = time.time()
        logger.log("\n\nPERFORMING TRISECTION NUMBER", counter+1)
        index_max = np.argmax(rectangle.dim_len)
        axes1, axes2, axes3 = copy.deepcopy(rectangle.axes), copy.deepcopy(rectangle.axes), copy.deepcopy(rectangle.axes)
        axes1[index_max][1] = axes1[index_max][0]+(axes1[index_max][1]-axes1[index_max][0])/3
        axes2[index_max][0], axes2[index_max][1] = axes2[index_max][0]+(axes2[index_max][1]-axes2[index_max][0])/3, axes2[index_max][0]+2*(axes2[index_max][1]-axes2[index_max][0])/3
        axes3[index_max][0] = axes3[index_max][0]+2*(axes3[index_max][1]-axes3[index_max][0])/3
        rectangle1, rectangle2, rectangle3  = Rectangle(rectangle.G, axes1, counter, direct_iteration, train_iterations, rectangles.index(rectangle), predictor), Rectangle(rectangle.G, axes2, counter, direct_iteration, train_iterations, rectangles.index(rectangle), predictor, True, rectangle), Rectangle(rectangle.G, axes3, counter, direct_iteration, train_iterations, rectangles.index(rectangle), predictor)
        rectangles[rectangles.index(rectangle)] = rectangle2 # inserting the middle rectangle in the place of the old, so to correspond with the other stats (training stats)
        rectangles.extend([rectangle1, rectangle3])
        counter += 1
        rectangles = renew_lists(rectangles, counter, indexes)
        end = time.time()
        logger.log("\nThe time it took for this trisection was", end-start, "secs.")
    return rectangles, counter, predictor


def renew_lists(rectangles, counter, indexes):
    for rec in rectangles:
        for i in range(1 + 2*counter - len(rec.rec_size_list)):
            rec.rec_size_list.extend([rec.size])
            rec.rec_diag_list.extend([rec.diag])  
            if rectangles.index(rec) in indexes:
                rec.optimal.extend([True])
            else:
                rec.optimal.extend([False])

        #for i in range(1*rec.iterations + 2*counter*rec.iterations - len(rec.train_size_list)):
         #   rec.train_size_list.extend([rec.size])
          #  rec.train_diag_list.extend([rec.diag])
    return rectangles
        
        
def draw_rectangles(rectangles):
    """    draws all the parameters compared to the 1st filter parameter. for ex. filter par 1-embed dim, filter par 1- filter par 2, filter par 1- filter par 3, filter par 1 - filter par 4"""
    centres, array = [[] for i in range(len(rectangles))], [[[] for i in range(rectangles[0].dims)] for j in range(len(rectangles))]
    for r,rectangle in enumerate(rectangles):       
        for i in range(rectangle.dims):
            if i == 0:
                array[r][i].extend([rectangle.axes[i][0], rectangle.axes[i][0], rectangle.axes[i][1], rectangle.axes[i][1], rectangle.axes[i][0]])
                centres[r].extend([rectangle.centre[i]])
            else:
                array[r][i].extend([rectangle.axes[i][0], rectangle.axes[i][1], rectangle.axes[i][1], rectangle.axes[i][0], rectangle.axes[i][0]])
                centres[r].extend([rectangle.centre[i]])

        fig2 = plt.figure(1, figsize=((10*(rectangle.dims)), 10))
        for i in range(0, rectangle.dims-1):      
            plt.subplot(1, int(rectangle.dims-1), i+1)
            plt.xlabel("filter parameter: 1")
            plt.ylabel("filter parameter:%s"%(i+2)) 
            plt.plot(array[r][0], array[r][i+1])
            plt.scatter(centres[r][0], centres[r][i+1])
            
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
    
   