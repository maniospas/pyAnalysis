import visualize.visualizer as vis
import graph_generation
import parameters
import main_operations as main_ops
import predictor as pred
import training
import matplotlib.pyplot as plt
import numpy as np
import math





G, module = graph_generation.create_pyan_graph("../TheAlgorithms-master", content = "predicates")
#vis.visualize(G)

pars = parameters.parameters(G)


predictor = pred.Predictor(False, svr_params=['rbf', 1e3, 1], test_predictor_acc=False)


#res = main_ops.cross_validate_svr()

         
#results_auc, results_sihl, results_loss, configurations, best_auc, best_sihl, best_loss, predictor = main_ops.find_best_settings(G, pars, predictor)


#rectangles, sorted_results, trisection_counter, predictor = main_ops.find_best_settings_direct(G=G, filter_dim=6, epsilon=10**(-4),
#                                                                                          stop_condition_perc=1, trisection_lim=50, predictor=predictor)


#pars2 = parameters.parameters(G, False, False, [], [], True, [[1, 0.5, 0.2]], [10], 1, math.inf)
#for i in range(5):
 #   training.train(G, pars2, predictor)
 
 
#pars = parameters.parameters(G, False, False, [], [], True, [[1, 0.7, 0.3]], [2], 1, math.inf)
#training.train(G, pars, predictor)


xt = [[[int(4+2**(x))]] for x in range(10)]
pars = [parameters.parameters(G, False, False, [], [], True, [[0.5, 0.7, 0.3]], [int(4+2**(x))], 1, math.inf) for x in range(10)]
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
plot7.set_xticklabels(xt) 
plot7.plot(x_plot, y_plot)





