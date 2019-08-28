import matplotlib.pyplot as plt
import numpy as np
import warnings
import logger


warnings.filterwarnings("ignore", module="matplotlib")

def compute_end_stats(predictor, show):          
    if predictor.predictor_on:
        training_added = [((i == 0) or (i == 3) or (i == 4)) for i in predictor.stats]
        training_added_plot = []
    
        trainings_cancelled = [((i == 1) or (i == 2)) for i in predictor.stats]
        
        acc_continuations = []
        for x in predictor.stats:
            if x == 3:
                acc_continuations.append(True)
            elif x == 4:
                acc_continuations.append(False)
            else:
                acc_continuations.append(np.nan)
        
        completed_runs = 0    
        for i,x  in enumerate(training_added):
            if x:
                training_added_plot.append(i)
                completed_runs += 1
        acc_cancellations = []
        if predictor.test_predictor_acc:
            for x in predictor.stats:
                if x == 1:
                    acc_cancellations.append(True)
                elif x == 2:
                    acc_cancellations.append(False)
                else:
                    acc_cancellations.append(np.nan)
        
        if show:            
            logger.log(str(completed_runs) + " full runs have completed out of " + str(len(predictor.stats)) + ". That is " + str(completed_runs/len(predictor.stats)*100) + " %\n")
            if predictor.test_predictor_acc:
                if trainings_cancelled.count(True):
                    logger.log("Out of total " + str(trainings_cancelled.count(True)) + " cancellations, " + str(acc_cancellations.count(True)) + " of them were correctly cancelled. That is " + str(acc_cancellations.count(True)/ trainings_cancelled.count(True)*100) + " %\n")
                else:
                    logger.log("No cancellations were made")
            if acc_continuations.count(True)+acc_continuations.count(False) != 0:
                logger.log("Out of total " + str(acc_continuations.count(True)+acc_continuations.count(False)) + " chosen continuations, " + str(acc_continuations.count(True)) + " of them were correctly continued. That is " + str(acc_continuations.count(True)/(acc_continuations.count(True)+acc_continuations.count(False))*100) + " %\n")
            else: 
                logger.log("No chosen continuations were made!")
        
        return training_added, training_added_plot, trainings_cancelled,  acc_continuations,  acc_cancellations



def plot_results(metric_to_plot, results, configs, predictor):        
    if metric_to_plot == "sihlouette":
        index = 0
    elif metric_to_plot == "auc":
        index = 1
    elif metric_to_plot == "loss":
        index = 2
    else:
        raise Exception("The metric that is requested to be plotted is not valid. The available metrics are: 'sihlouette', 'auc', and 'loss'.")
                
    fig1 = plt.figure(1, figsize=(len(predictor.stats), 35 if predictor.predictor_on else 7))
    
    plot7 =  plt.subplot(15 if predictor.predictor_on else 3, 1, 1)
    x_plot = predictor.loss_list
    y_plot = [0 for x in x_plot]
    x_plot2 = [np.mean(predictor.loss_list) + np.std(predictor.loss_list), np.mean(predictor.loss_list) + 2*np.std(predictor.loss_list), np.mean(predictor.loss_list) + 3*np.std(predictor.loss_list), np.mean(predictor.loss_list) - np.std(predictor.loss_list), np.mean(predictor.loss_list) - 2*np.std(predictor.loss_list), np.mean(predictor.loss_list) - 3*np.std(predictor.loss_list)]
    y_plot2 = [0 for x in x_plot2]
    x_plot3 = [np.mean(predictor.loss_list)]
    y_plot3 = [0 for x in x_plot3]
    plot7.set_xlabel("losses scattering")
    plot7.scatter(x_plot, y_plot, marker="^", c="blue")
    plot7.scatter(x_plot2, y_plot2, marker="+", c="red")
    plot7.scatter(x_plot3, y_plot3, marker="+", c="yellow")
                
    plot5 = plt.subplot(13 if predictor.predictor_on else 3, 1, 3)
    y_plot = [abs(i) for i in results[index]]
    x_plot = [i for i in range(len(y_plot))]
    plot5.set_ylabel("loss")
    plot5.set_xlabel("experiment configuration")
    plot_lim = max(y_plot)
    plot5.set_ylim([0, 0.03])
    plot5.scatter(x_plot, y_plot)
    plot5.plot([i for i in x_plot], [0 for i in range(len(x_plot))], c="black")
    plot5.set_xticks(np.arange(len(x_plot))) 
    plot5.set_xticklabels([(j, [round(k,3) for k in i[0]], i[1]) for j,i in enumerate(configs)])  
        
    if predictor.predictor_on:
        
        training_added, training_added_plot, trainings_cancelled, acc_continuations, acc_cancellations = compute_end_stats(predictor, True)
                            
        plot1 = plt.subplot(13, 1, 5)
        plot1.set_title("Intermediate errors")
        plot1.set_ylabel("intermediate errors")
        plot1.set_xlabel("experiment configuration")        
        y_plot = [abs(i) for i in predictor.int_errors]
        plot_lim = max(y_plot[10:])
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
        plot1.plot([i for i in x_plot], [0.0015 for i in range(len(x_plot))], c="black")
        plot1.set_xticks(np.arange(len(x_plot)))            
        plot1.set_xticklabels([(k+1, [round(j,3) for j in i[0]], i[1]) for k,i in enumerate(configs)]) 
             
        plot4 = plt.subplot(13, 1, 7)
        plot4.set_title("Intermediate errors percentage")
        plot4.set_ylabel("intermediate errors percentage")
        plot4.set_xlabel("experiment configuration")        
        y_plot = [abs(i) for i in predictor.int_errors_perc]
        plot_lim = max(y_plot[10:])
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
        
        plot2 = plt.subplot(13, 1, 9)
        plot2.set_title("Final errors")
        plot2.set_ylabel("final errors")
        plot2.set_xlabel("experiment configuration")
        y_plot = [abs(i) for i in predictor.f_errors]
        a = [i for i in y_plot if not np.isnan(i)][10:]
        plot_lim = max(a) if (len(a) != 0) else 0.005
        plot2.set_ylim(0, plot_lim)
        x_plot = [i for i in range(len(y_plot))]
        plot2.scatter(x_plot, y_plot)
        plot2.plot([i for i in x_plot], [0.0015 for i in range(len(x_plot))], c="black")
        plot2.set_xticks(np.arange(len(x_plot)))
        plot2.set_xticklabels([(k+1, [round(j,3) for j in i[0]], i[1]) for k,i in enumerate(configs)])              

        plot6 = plt.subplot(13, 1, 11)
        plot6.set_title("Final errors percentage")
        plot6.set_ylabel("final errors percentage")
        plot6.set_xlabel("experiment configuration")
        y_plot = [abs(i) for i in predictor.f_errors_perc]
        a = [i for i in y_plot if not np.isnan(i)][10:]
        plot_lim = max(a) if len(a)!= 0 else 50
        plot6.set_ylim(0, plot_lim)
        x_plot = [i for i in range(len(y_plot))]
        plot6.scatter(x_plot, y_plot)
        plot6.plot([i for i in x_plot], [10 for i in range(len(x_plot))], c="black")
        plot6.set_xticks(np.arange(len(x_plot)))
        plot6.set_xticklabels([(k+1, [round(j,3) for j in i[0]], i[1]) for k,i in enumerate(configs)])   
                
        plot3 = plt.subplot(13, 1, 13)
        plot3.set_title("Percentage of cancellations every iteration" + " - Percentage of succesfull cancellations very iteration" if predictor.test_predictor_acc else "")
        plot3.set_ylabel("Percentage of cancellations every iteration" + " - Percentage of succesfull cancellations very iteration" if predictor.test_predictor_acc else "")
        plot3.set_xlabel("experiment number")
        plot3.set_ylim(0, 100)
        canc_count, acc_canc_count, trainings_cancelled_sums, acc_canc_sums = 0, 0, [], []
        for x in trainings_cancelled:
            if x == True:
                canc_count += 1
            trainings_cancelled_sums.append(canc_count)
        for l in acc_cancellations:
            if l == True:
                acc_canc_count += 1
            acc_canc_sums.append(acc_canc_count)
        x_plot = [i+1 for i in range(len(trainings_cancelled_sums))]
        y_plot = [(i/j)*(100) for i,j in zip(trainings_cancelled_sums, x_plot)]        
        plot3.plot(x_plot, y_plot)
        if predictor.test_predictor_acc:
            y_plot = [(i/j)*(100) for i,j in zip(acc_canc_sums, x_plot)]
            plot3.plot(x_plot, y_plot)
        
    return fig1

        
