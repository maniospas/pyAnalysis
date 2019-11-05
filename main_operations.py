import training
import numpy as np
import logger
import plots
import direct
import matplotlib.pyplot as plt
import results_to_csv
from sklearn.svm import SVR
import math_operations as math_ops
import math
import time
import graph_generation
import parameters
import predictor as pred




def find_best_settings_direct(G, filter_dim, epsilon, trisection_lim, train_iterations, predictor):  
    if filter_dim < 1: raise Exception("The filter dimensions cannot be less than 1!")

    starting_filter_parameters = [[0,1] for x in range(int(filter_dim))]
    rectangles = [direct.Rectangle(G, starting_filter_parameters, -1, 0, train_iterations, np.nan, predictor)]
    counter, direct_iteration  = 0, 1

    start = time.time()    
    while True:
        indexes = direct.find_optimal_rectangles(rectangles, epsilon)
        rectangles, counter, predictor = direct.trisect(indexes, rectangles, counter, direct_iteration, trisection_lim, train_iterations, predictor)
        direct_iteration += 1
        if counter >= trisection_lim: break
        fig = direct.draw_rectangles(rectangles)
    end = time.time()
    logger.log("The time it took for the DIRECT algorithm in total was", end-start, "secs or", (end-start)/60, "hours.")

    fig = direct.draw_rectangles(rectangles)
        
    logger.log(counter, "trisections have been performed.")
    
    #logger.log("The lowest loss value was", loss_min, "and was found at the following filter:", best_rectangle.centre, ". At this filter, the sihlouette value was", best_rectangle.sihl)

    rec_results, rec_configs, train_results, train_configs = list_results(rectangles)
    
    fig2 = plots.plot_results(train_results, train_configs, rectangles, predictor)
    plt.show
    
    sorted_results = [direct.sort_rectangles(rectangles, "loss"), direct.sort_rectangles(rectangles, "sihlouette")] 
    
    if predictor.test_predictor_acc: results_to_csv.cvalidation_data_to_csv(predictor.cvalidation_data)
    results_to_csv.export_results_to_csv(rec_results, rec_configs, train_results, train_configs, predictor, True, rectangles, sorted_results)
    
    return rectangles, sorted_results, counter, predictor


def cross_validate_svr():    
    kernels = ["linear", "rbf", "poly", "sigmoid"]
    cs = [2**(x) for x in range(-10, 10, 1)]
    gammas = [2**(x) for x in range(-10, 10, 1)]
    params = []
    for kernel in kernels:
        for c in cs:
            for gamma in gammas:
                params.append([kernel, c, gamma])

    errors = [[] for i in range(len(params))]
    cval_data = results_to_csv.access_cvalidation_data()
    
    for j in range(3):
        train_data = [x for (i,x) in enumerate(cval_data[1:]) if i%3==j]                 
        test_data = [x for (i,x) in enumerate(cval_data[1:]) if i%3!=j]                 
    
        for i,par in enumerate(params):
            i_errors, f_errors = [], []
            for row in test_data:
                i_errors.append(abs(math.exp(SVR(kernel=par[0], C=par[1], gamma=par[2]).fit(math_ops.transpose(train_data)[0], math_ops.transpose(train_data)[1]).predict([row[0]])[0]) - math.exp(row[1])))    
                f_errors.append(abs(math.exp(SVR(kernel=par[0], C=par[1], gamma=par[2]).fit(math_ops.transpose(train_data)[0], math_ops.transpose(train_data)[2]).predict([row[0]])[0]) - math.exp(row[2])))    
            if i%100==0: logger.log(i, "iteration completed out of", len(params)) 
            errors[i].append([sum(i_errors)/len(i_errors), sum(f_errors)/len(f_errors), par])
            
    errors_sum = [[(x[0][0]+x[1][0]+x[2][0])/3, (x[0][1]+x[1][1]+x[2][1])/3, x[0][2]] for x in errors]
    
    logger.log("the minimum intermediate error was found with", params[[x[0] for x in errors_sum].index(min([x[0] for x in errors_sum]))], "and is", min([x[0] for x in errors_sum]))
    logger.log("the minimum final error was found with", params[[x[1] for x in errors_sum].index(min([x[1] for x in errors_sum]))], "and is", min([x[1] for x in errors_sum]))    
    #logger.log(errors)
    
    return errors

    
def list_results(rectangles):
    rec_loss, rec_sihl, rec_configs, train_loss, train_sihl, train_configs = [], [], [], [], [], []
    for rec in rectangles:
        train_loss.extend(rec.losses)
        train_sihl.extend(rec.sihls)
        train_configs.extend([rec.configs[0] for x in range(rec.iterations)])
        rec_loss.append(rec.loss)
        rec_sihl.append(rec.sihl)
        rec_configs.append(rec.configs[0])
    return [(rec_sihl), (rec_loss)], rec_configs, [(train_sihl), (train_loss)], train_configs

                
def m_function(software):

    G = graph_generation.create_predicate_graph(software)
    #vis.visualize(G)
    
    pars = parameters.parameters(G)
    
    import predictor as pred
    predictor = pred.Predictor(True, svr_params=['rbf', 1000, 1, 0.01], test_predictor_acc=False)
    

    rectangles, sorted_results, trisection_counter, predictor = find_best_settings_direct(G=G, filter_dim=5, epsilon=10**(-4),
                                                        trisection_lim=70, train_iterations=3, predictor=predictor)
    return rectangles, sorted_results, trisection_counter, predictor
    
    
    #res = main_ops.cross_validate_svr()
     
    #pars = parameters.parameters(G, False, False, [], [], True, [[0.16, 0.83, 0.83, 0.83, 0.16]], [32 for i in range(1)], 1, 2200)
    #res, conf, pred = training.train(G, pars, predictor)


    """
    pars = [parameters.parameters(G, False, False, [], [], True, [[0.5, 0.7, 0.3, 0.5, 0.5, 0.1, 0.5, 0.1]], [100], 1, math.inf) for x in range(5)]
    results = []
    x_plot = []
    configs = []
    for par in pars:
        res, conf, pred = training.train(G, par, predictor)
        results.append(res[2])
        x_plot.append(par)
    
    fig1 = plt.figure(1, figsize=(10,10))
    
    plot7 =  plt.subplot(15 if predictor.predictor_on else 3, 1, 1)
    y_plot = results
    x_plot = [i for i in range(len(y_plot))]
    plot7.set_xticks(np.arange(len(x_plot)))            
    plot7.plot(x_plot, y_plot)
    """
    """
    results_sum = []
    for j in range(5):
        pars = [parameters.parameters(G, False, False, [], [], True, [[0.9, 0.5, 0.8, 0.5, 0.5, 0.1, 0.5, 0.1]], [100], 1, math.inf) for x in range(5)]
        results = []
        x_plot = []
        configs = []
        for par in pars:
            res, conf, pred = training.train(G, par, predictor)
            results.append(res[2])
            x_plot.append(par)
        results_sum.append(sum(results)/len(results))
    
    fig1 = plt.figure(1, figsize=(10,10))
    
    plot7 =  plt.subplot(15 if predictor.predictor_on else 3, 1, 1)
    y_plot = results_sum
    x_plot = [i for i in range(len(y_plot))]
    plot7.set_xticks(np.arange(len(x_plot)))            
    plot7.plot(x_plot, y_plot)
    """
    

