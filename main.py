import visualize.visualizer as vis
import graph_generation
import parameters
import main_operations as mo
import predictor as pred
#import pygame




G, module = graph_generation.create_pyan_graph("../TheAlgorithms-master", content = "predicates")
#vis.visualize(G)

pars = parameters.parameters(G)

predictor = pred.predictor(True, "svr", svr_params=['rbf', 1e3, 1], arima_params=[2, 0, 2], arma_params=[1, 1], test_predictor_acc=False)


#intermediate_res, final_res = mo.cross_validate_svr()

         
#results_auc, results_sihl, results_loss, configurations, best_auc, best_sihl, best_loss, predictor = mo.find_best_settings(G, pars, predictor)


rectangles, trisection_counter, predictor = mo.find_best_settings_direct(G, 0.1, 100, predictor)




