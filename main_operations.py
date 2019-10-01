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


def find_best_settings(G, pars, predictor):       
    results, configs, predictor = training.train(G, pars, predictor)
    plots.plot_results(results, configs, predictor)
    results_to_csv.export_results_to_csv("loss", results, configs, predictor, False, [])
    max_auc, max_sihl, min_loss = np.amax(results[0]), np.amax(results[1]), np.amin(results[2])
    from numpy import unravel_index
    max_auc_index = unravel_index(results[0].argmax(), results[0].shape)
    max_sihl_index = unravel_index(results[1].argmax(), results[1].shape)
    min_loss_index = unravel_index(results[2].argmin(), results[2].shape)
    
    logger.log("The best AUC result was", max_auc,"and was found when using the following filter", configs[max_auc_index[0]][0],
           "and with embedding dimensions", configs[max_auc_index[0]][1], ". The Sihlouette result at these settings was",
           results[1][max_auc_index[0]], "and the Loss was", results[2][max_auc_index[0]] , ".\n\n")
    logger.log("The best Sihlouette result was", max_sihl ," and was found when using the following filter", configs[max_sihl_index[0]][0],
          "and with embedding dimensions", configs[max_sihl_index[0]][1], ". The AUC result at these settings was", results[0][max_sihl_index[0]],
          "and the Loss was", results[2][max_sihl_index[0]], ".\n\n")
    logger.log("The best Loss result was", min_loss ," and was found when using the following filter", configs[min_loss_index[0]][0], 
          "and with embedding dimensions", configs[min_loss_index[0]][1], ". The AUC result at these settings was", results[0][min_loss_index[0]],
          "and the Sihlouette was", results[1][min_loss_index[0]], ".\n\n")  
           
    return results[0], results[1], results[2], configs, max_auc_index, max_sihl_index, min_loss_index, predictor


def find_best_settings_direct(G, filter_dim, epsilon, trisection_lim, train_iterations, predictor):  
    if filter_dim < 1: raise Exception("The filter dimensions cannot be less than 1!")
    
    rectangles = []
    starting_filter_parameters = [[0,1] for x in range(int(filter_dim))]
    rectangles.append(direct.Rectangle(G, starting_filter_parameters, -1, 0, train_iterations, np.nan, predictor))
    counter, direct_iteration, optimal_results = 0, 1, []

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
    
    if predictor.test_predictor_acc: results_to_csv.cvalidation_data_to_csv(predictor.cvalidation_data)
    results_to_csv.export_results_to_csv(rec_results, rec_configs, train_results, train_configs, predictor, True, rectangles)

    sorted_results = direct.sort_rectangles(rectangles, "loss")       
    
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
    rec_loss, rec_sihl, rec_auc, rec_configs, train_loss, train_sihl, train_auc, train_configs = [], [], [], [], [], [], [], []
    for rec in rectangles:
        train_loss.extend(rec.losses)
        train_sihl.extend(rec.sihls)
        train_auc.extend(rec.aucs)
        train_configs.extend([rec.configs[0] for x in range(rec.iterations)])
        rec_loss.append(rec.loss)
        rec_sihl.append(rec.sihl)
        rec_auc.append(rec.auc)
        rec_configs.append(rec.configs[0])
    return [(rec_sihl), (rec_auc), (rec_loss)], rec_configs, [(train_sihl), (train_auc), (train_loss)], train_configs

                
    

