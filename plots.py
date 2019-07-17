import matplotlib.pyplot as plt
import numpy as np
import logger
import warnings


warnings.filterwarnings("ignore", module="matplotlib")



def plot_results(metric_to_plot, results, configs, predictor):
        
    if metric_to_plot == "sihlouette":
        index = 0
    elif metric_to_plot == "auc":
        index = 1
    elif metric_to_plot == "loss":
        index = 2
    else:
        raise Exception("The metric that is requested to be plotted is not valid. The available metrics are: 'sihlouette', 'auc', and 'loss'.")
        
    total_runs = len(predictor.stats)
        
    fig1 = plt.figure(1, figsize=(total_runs, 35 if predictor.predictor_on else 7)) 
                
    plot5 = plt.subplot(11 if predictor.predictor_on else 1, 1, 1)
    #plot1.set_title(metric, "for each configuration")
    y_plot = [abs(i) for i in results[index]]
    x_plot = [i for i in range(len(y_plot))]
    plot5.set_ylabel(metric_to_plot)
    plot5.set_xlabel("experiment configuration")
    plot5.set_ylim([0, 0.03])
    plot5.scatter(x_plot, y_plot)
    plot5.plot([i for i in x_plot], [0 for i in range(len(x_plot))], c="black")
    plot5.set_xticks(np.arange(len(x_plot))) 
    plot5.set_xticklabels([(j, [round(k,3) for k in i[0]], i[1]) for j,i in enumerate(configs)])  
        
    if predictor.predictor_on:
        
        training_added_plot, trainings_cancelled_sum = predictor.compute_end_stats()
                            
        plot1 = plt.subplot(11, 1, 3)
        plot1.set_title("Intermediate errors")
        plot1.set_ylabel("intermediate errors")
        plot1.set_xlabel("experiment configuration")        
        y_plot = [abs(i) for i in predictor.int_errors]
        plot_lim = max(y_plot[1:])
        plot1.set_ylim([0, plot_lim])
        x_plot = [i for i in range(len(y_plot))]
        plot1.plot(x_plot, y_plot)        
        y_plot_trained = [np.nan for i in range(len(x_plot))]
        for i in training_added_plot:
            a = y_plot[i]
            y_plot[i] = np.nan
            y_plot_trained[i] = a        
        plot1.scatter(x_plot, y_plot, marker="^", c="blue")
        plot1.scatter(x_plot, y_plot_trained, marker="+", c="red")
        plot1.plot([i for i in x_plot], [0.001 for i in range(len(x_plot))], c="black")
        plot1.set_xticks(np.arange(len(x_plot)))            
        plot1.set_xticklabels([(k+1, [round(j,3) for j in i[0]], i[1]) for k,i in enumerate(configs)]) 
             
        plot4 = plt.subplot(11, 1, 5)
        plot4.set_title("Intermediate errors percentage")
        plot4.set_ylabel("intermediate errors percentage")
        plot4.set_xlabel("experiment configuration")        
        y_plot = [abs(i) for i in predictor.int_errors_perc]
        plot_lim = max(y_plot[1:])
        plot4.set_ylim([0, plot_lim])
        x_plot = [i for i in range(len(y_plot))]
        plot4.plot(x_plot, y_plot)        
        y_plot_trained = [np.nan for i in range(len(x_plot))]
        for i in training_added_plot:
            a = y_plot[i]
            y_plot[i] = np.nan
            y_plot_trained[i] = a  
        plot4.scatter(x_plot, y_plot, marker="^", c="blue")
        plot4.scatter(x_plot, y_plot_trained, marker="+", c="red")        
        plot4.plot([i for i in x_plot], [10 for i in range(len(x_plot))], c="black")
        plot4.set_xticks(np.arange(len(x_plot)))
        plot4.set_xticklabels([(k+1, [round(j,3) for j in i[0]], i[1]) for k,i in enumerate(configs)])            
        
        plot2 = plt.subplot(11, 1, 7)
        plot2.set_title("Final errors")
        plot2.set_ylabel("final errors")
        plot2.set_xlabel("experiment configuration")
        y_plot = [abs(i) for i in predictor.f_errors]
        a = [i for i in y_plot if not np.isnan(i)][1:]
        plot_lim = max(a) if (len(a) != 0) else 0.005
        plot2.set_ylim(0, plot_lim)
        x_plot = [i for i in range(len(y_plot))]
        plot2.scatter(x_plot, y_plot)
        #y_plot_trained = [np.nan for i in range(len(x_plot))]
        #for i in training_added_plot:
            #a = y_plot[i]
            #y_plot[i] = np.nan
            #y_plot_trained[i] = a  
        #plot4.scatter(x_plot, y_plot, marker="^", c="blue")
        #plot4.scatter(x_plot, y_plot_trained, marker="+", c="red")         
        plot2.plot([i for i in x_plot], [0.001 for i in range(len(x_plot))], c="black")
        plot2.set_xticks(np.arange(len(x_plot)))
        plot2.set_xticklabels([(k+1, [round(j,3) for j in i[0]], i[1]) for k,i in enumerate(configs)])              

        plot6 = plt.subplot(11, 1, 9)
        plot6.set_title("Final errors percentage")
        plot6.set_ylabel("final errors percentage")
        plot6.set_xlabel("experiment configuration")
        y_plot = [abs(i) for i in predictor.f_errors_perc]
        a = [i for i in y_plot if not np.isnan(i)][1:]
        plot_lim = max(a) if len(a)!= 0 else 50
        plot6.set_ylim(0, plot_lim)
        x_plot = [i for i in range(len(y_plot))]
        plot6.scatter(x_plot, y_plot)
        #y_plot_trained = [np.nan for i in range(len(x_plot))]
        #for i in training_added_plot:
            #a = y_plot[i]
            #y_plot[i] = np.nan
            #y_plot_trained[i] = a  
        #plot4.scatter(x_plot, y_plot, marker="^", c="blue")
        #plot4.scatter(x_plot, y_plot_trained, marker="+", c="red") 
        plot6.plot([i for i in x_plot], [10 for i in range(len(x_plot))], c="black")
        plot6.set_xticks(np.arange(len(x_plot)))
        plot6.set_xticklabels([(k+1, [round(j,3) for j in i[0]], i[1]) for k,i in enumerate(configs)])   
                
        plot3 = plt.subplot(11, 1, 11)
        plot3.set_title("Percentage of cancellations every iteration")
        plot3.set_ylabel("Percentage of cancellations every iteration")
        plot3.set_xlabel("experiment number")
        plot3.set_ylim(0, 100)        
        x_plot = [i+1 for i in range(len(trainings_cancelled_sum))]
        y_plot = [(i/j)*(100) for i,j in zip(trainings_cancelled_sum, x_plot)]        
        plot3.plot(x_plot, y_plot)
        #"""for i in trainings_cancelled_plot:
         #   plot3.plot([i, i], [0, predictor.total_runs], c="orange")"""

        logger.log(predictor.completed_runs, "full runs have completed out of", total_runs)
        
    return fig1

        
