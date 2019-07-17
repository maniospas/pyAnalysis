import random
import math_operations as mo
import numpy as np


class parameters:
    def __init__(self, G, default_on=True, templates_on=True, filter_types=[], filter_depths=[], custom_on=False, custom_filter_pars=[], emb_dims=[], iterations=1, epochs=1):
        #self.perform_value_check(selection_type, default_or_custom, filter_types, filter_depths, emb_dims, iterations, epochs)
        self.filters = []
        self.emb_dims = emb_dims
        self.iterations = iterations
        self.epochs = epochs
        if default_on:
            filter_types = ["uniform", "exponential", "heat", "linear", "polynomial"]
            # all available filter types: "uniform", "last", "exponential", "heat", "linear", "polynomial"
            filter_depths = [i for i in range(1,4)]
            dim_algorithm = lambda x: 2**(x)
            self.emb_dims = [dim_algorithm(i) for i in range(2,4)]
            self.iterations = 1
            self.epochs = 5000
            custom_filter_pars = []
        if templates_on:        
            for f in filter_types:
                for d in filter_depths:
                    self.filters.append(mo.create_filter(f, d))
            self.filters = remove_duplicates(self.filters)
        if custom_on:
            for filt in custom_filter_pars:    
                self.filters.append(filt)
            
    def random_values(self, selection, number, depth):
        parameters_list = []
        span = [0, 1]
        while len(parameters_list) < number:
            rand_pars = []
            depth = int(random.random(1, 7))
            for i in range(depth): rand_pars.append(random.random(span[0], span[1]))
            if rand_pars not in parameters_list: parameters_list.append(rand_pars)
        return parameters_list
    
def remove_duplicates(duplicate_list): 
    final_list = [] 
    for num in duplicate_list: 
        if num not in final_list: 
            final_list.append(num)
        #else: print("duplicate:", num)
    return final_list 