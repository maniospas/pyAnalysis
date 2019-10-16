import matplotlib.pyplot as plt
import numpy as np
import warnings
import logger
import math


warnings.filterwarnings("ignore", module="matplotlib")

def compute_end_stats(predictor, show):          
    if predictor.predictor_on:
        training_added, training_added_plot, trainings_cancelled, acc_continuations, completed_runs, acc_cancellations = [((i == 0) or (i == 3) or (i == 4)) for i in predictor.stats], [], [((i == 1) or (i == 2)) for i in predictor.stats], [], 0, []

        for x in predictor.stats:
            if x == 3:
                acc_continuations.append(True)
            elif x == 4:
                acc_continuations.append(False)
            else:
                acc_continuations.append(np.nan)
         
        for i,x  in enumerate(training_added):
            if x:
                training_added_plot.append(i)
                completed_runs += 1

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



def plot_results(results, configs, rectangles, predictor):        
                
    fig1 = plt.figure(1, figsize=(len(predictor.stats), 35 if predictor.predictor_on else 7))
    
    plot7 =  plt.subplot(15 if predictor.predictor_on else 5, 1, 1)
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
                
    plot5 = plt.subplot(15 if predictor.predictor_on else 5, 1, 3)
    y_plot = [abs(i) for i in results[1]]
    x_plot = [i for i in range(len(y_plot))]
    plot5.set_ylabel("loss")
    plot5.set_xlabel("experiment configuration")
    plot_lim = max([x for x in y_plot if (x is not math.inf)])
    #plot5.set_ylim([0.12, plot_lim])
    plot5.scatter(x_plot, y_plot)
    plot5.plot([i for i in x_plot], [0 for i in range(len(x_plot))], c="black")
    plot5.set_xticks(np.arange(len(x_plot))) 
    a = configs
    plot5.set_xticklabels([(j, [round(k,3) for k in i[0]], i[1]) for j,i in enumerate(configs)])  
    
    plot8 = plt.subplot(15 if predictor.predictor_on else 5, 1, 5)
    rec = rectangles[[x.loss for x in rectangles].index(min([x.loss for x in rectangles]))]
    y_plot, x_ticks = [], []
    while True:
        y_plot.append(rec.loss)
        x_ticks.append(rec.configs)
        if rec.parent_index is np.nan: break
        rec = rectangles[rec.parent_index]        
    y_plot, x_ticks = y_plot[::-1], x_ticks[::-1]
    x_plot = [i for i in range(len(y_plot))]
    plot8.set_ylabel("path of best loss")
    plot8.plot(x_plot, y_plot)
    plot8.set_xticks(np.arange(len(x_plot))) 
    plot8.set_xticklabels(x_ticks)  

        
    if predictor.predictor_on:
        
        training_added, training_added_plot, trainings_cancelled, acc_continuations, acc_cancellations = compute_end_stats(predictor, True)
                            
        plot1 = plt.subplot(15, 1, 7)
        plot1.set_title("Intermediate errors")
        plot1.set_ylabel("intermediate errors")
        plot1.set_xlabel("experiment configuration")        
        y_plot = [abs(i) for i in predictor.int_errors]
        plot_lim = max(y_plot[10:]) if len(y_plot)>10 else 100
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
        plot1.plot([i for i in x_plot], [0.01 for i in range(len(x_plot))], c="black")
        plot1.set_xticks(np.arange(len(x_plot)))            
        plot1.set_xticklabels([(k+1, [round(j,3) for j in i[0]], i[1]) for k,i in enumerate(configs)]) 
             
        plot4 = plt.subplot(15, 1, 9)
        plot4.set_title("Intermediate errors percentage")
        plot4.set_ylabel("intermediate errors percentage")
        plot4.set_xlabel("experiment configuration")        
        y_plot = [abs(i) for i in predictor.int_errors_perc]
        plot_lim = max(y_plot[10:]) if len(y_plot)>10 else 50
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
        plot4.plot([i for i in x_plot], [5 for i in range(len(x_plot))], c="black")
        plot4.set_xticks(np.arange(len(x_plot)))
        plot4.set_xticklabels([(k+1, [round(j,3) for j in i[0]], i[1]) for k,i in enumerate(configs)])            
        
        plot2 = plt.subplot(15, 1, 11)
        plot2.set_title("Final errors")
        plot2.set_ylabel("final errors")
        plot2.set_xlabel("experiment configuration")
        y_plot = [abs(i) for i in predictor.f_errors]
        a = [i for i in y_plot if not np.isnan(i)][10:]
        plot_lim = max(a) if (len(a) != 0) else 100
        plot2.set_ylim(0, plot_lim)
        x_plot = [i for i in range(len(y_plot))]
        plot2.scatter(x_plot, y_plot)
        plot2.plot([i for i in x_plot], [0.01 for i in range(len(x_plot))], c="black")
        plot2.set_xticks(np.arange(len(x_plot)))
        plot2.set_xticklabels([(k+1, [round(j,3) for j in i[0]], i[1]) for k,i in enumerate(configs)])              

        plot6 = plt.subplot(15, 1, 13)
        plot6.set_title("Final errors percentage")
        plot6.set_ylabel("final errors percentage")
        plot6.set_xlabel("experiment configuration")
        y_plot = [abs(i) for i in predictor.f_errors_perc]
        a = [i for i in y_plot if not np.isnan(i)][10:]
        plot_lim = max(a) if len(a)!= 0 else 100
        plot6.set_ylim(0, plot_lim)
        x_plot = [i for i in range(len(y_plot))]
        plot6.scatter(x_plot, y_plot)
        plot6.plot([i for i in x_plot], [5 for i in range(len(x_plot))], c="black")
        plot6.set_xticks(np.arange(len(x_plot)))
        plot6.set_xticklabels([(k+1, [round(j,3) for j in i[0]], i[1]) for k,i in enumerate(configs)])   
                
        plot3 = plt.subplot(15, 1, 15)
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

        
