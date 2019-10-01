import visualize.visualizer as vis
import graph_generation
import parameters
import main_operations as main_ops
import predictor as pred
import training
import matplotlib.pyplot as plt
import numpy as np
import math





G, module = graph_generation.create_pyan_graph("../networkx", content = "predicates")
#vis.visualize(G)

pars = parameters.parameters(G)


predictor = pred.Predictor(True, svr_params=['rbf', 10000, 10], test_predictor_acc=False)

rectangles, sorted_results, trisection_counter, predictor = main_ops.find_best_settings_direct(G=G, filter_dim=8, epsilon=10**(-4),
                                                trisection_lim=70, train_iterations=3, predictor=predictor)


#res = main_ops.cross_validate_svr()
 
#pars = parameters.parameters(G, False, False, [], [], True, [[0.7, 0.5, 0.8, 0.3, 0.2, 0.8, 0.15, 0.2]], [100], 1, math.inf)
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
