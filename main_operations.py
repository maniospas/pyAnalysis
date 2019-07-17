import training
import numpy as np
import logger
import predictor as pred
import plots
import direct
import matplotlib.pyplot as plt
import parameters
import graph_generation
import math
import results_to_csv



def find_best_settings(G, pars, predictor):
       
    results, configs, predictor = training.train(G, pars, predictor)
    #plots.plot_results("loss", results, configs, predictor)
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


def find_best_settings_direct(G, k, trisection_lim, predictor):
    
    rectangles = []
    starting_filter_parameters = [[0,1], [0,1], [0,1], [0,1], [0,1], [0,1]]
    rectangles.append(direct.Rectangle(G, starting_filter_parameters, predictor))
    counter = 0
    
    while True:
        indexes = direct.find_optimal_rectangles(rectangles, k)
        if len(indexes) == 0 or (counter >= trisection_lim): break
        rectangles, counter, predictor = direct.trisect(indexes, rectangles, counter, predictor)

    fig = direct.draw_rectangles_1(rectangles)
        
    logger.log(counter, "trisections have been performed.")
    
    #logger.log("The lowest loss value was", loss_min, "and was found at the following filter:", best_rectangle.centre, ". At this filter, the sihlouette value was", best_rectangle.sihl)

    loss, sihl, auc, configs = [], [], [], []
    for rec in rectangles:
        loss.append(rec.loss)
        sihl.append(rec.sihl)
        auc.append(rec.auc)
        configs.append(rec.configs[0])
    results = [(sihl), (auc), (loss)]
    
    fig2 = plots.plot_results("loss", results, configs, predictor)
    plt.show
    
    results_to_csv.export_results_to_csv("loss", results, configs, predictor, True, rectangles)
   
    #rectangles, metrics, sorted_filters = direct.sort_rectangles(rectangles, "loss")    
    
    return rectangles, counter, predictor


def cross_validate_svr():
    
    kernels = ["rbf"]
    cs = [2**(x) for x in range(-10, 5, 3)]
    gammas = [2**(x) for x in range(-5, 10, 3)]
    params = []
    for kernel in kernels:
        for c in cs:
            for gamma in gammas:
                params.append([kernel, c, gamma])
                
    G, module = graph_generation.create_pyan_graph("../TheAlgorithms-master", content = "predicates")                
    pars = parameters.parameters(G)
    
    results = []            
    for i,par in enumerate(params):
        predictor = pred.predictor(True, "svr", svr_params=par, test_predictor_acc=True)
        results_auc, results_sihl, results_loss, configs, best_auc, best_sihl, best_loss, predictor = find_best_settings(G, pars, predictor)
        predictor.compute_end_stats()
        results.append([predictor.return_intermediate_errors_mean(), predictor.return_final_errors_mean(), predictor.completed_runs, par])
        print("results:", results)
        print("\n\n", i, "CROSS VALIDATION CYCLE COMPLETED OUT OF", len(params), "\n\n")
    
    results = np.asarray(results)
    results_intermediate = results[results[:,0].argsort()]
    results_final = results[results[:,1].argsort()]
    results_completed_runs = results[results[:,2].argsort()]
    logger.log("results sorted by intermediate prediction error means", results_intermediate, "\n")  
    logger.log("results sorted for final prediction error means", results_final, "\n")  
    logger.log("results sorted for total completed runs", results_completed_runs, "\n")
    
    
    return results_intermediate, results_final

                
    

