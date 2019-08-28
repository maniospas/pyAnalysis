import csv
import os
import logger
import plots



def export_results_to_csv(metric_to_plot, results, configs, predictor, direct_on=False, rectangles=[]):    
    if metric_to_plot == "sihlouette":
        index = 0
    elif metric_to_plot == "auc":
        index = 1
    elif metric_to_plot == "loss":
        index = 2
    else:
        raise Exception("The metric that is requested to be plotted is not valid. The available metrics are: 'sihlouette', 'auc', and 'loss'.")
        
    y_plot_metric = [abs(i) for i in results[index]]
    x_plot = [i for i in range(len(y_plot_metric))]
    x_ticks_configs = [(j, [round(k,3) for k in i[0]], i[1]) for j,i in enumerate(configs)]
    
    if predictor.predictor_on:
        
        training_added, training_added_plot, trainings_cancelled, acc_continuations, acc_cancellations = plots.compute_end_stats(predictor, False)
                
        y_plot_int_errors, y_plot_int_errors_perc, y_plot_f_errors, y_plot_f_errors_perc  = [abs(i) for i in predictor.int_errors],  [abs(i) for i in predictor.int_errors_perc], [abs(i) for i in predictor.f_errors], [abs(i) for i in predictor.f_errors_perc]

        create_csv(predictor, direct_on, x_plot, x_ticks_configs, y_plot_metric, y_plot_int_errors, y_plot_int_errors_perc, y_plot_f_errors,
               y_plot_f_errors_perc, training_added, trainings_cancelled, acc_continuations, acc_cancellations, rectangles) 
    else:
        create_csv(predictor, direct_on, x_plot, x_ticks_configs, y_plot_metric)

    
def create_csv(predictor, direct_on, x_plot, x_ticks_configs, y_plot_metric, y_plot_int_errors, y_plot_int_errors_perc, y_plot_f_errors,
               y_plot_f_errors_perc, training_added, trainings_cancelled, acc_continuations, acc_cancellations, rectangles):
    
    try:
        os.remove("plot_data.csv")
    except:
        logger.log("no file 'plot_data' to delete")
    
    csv_data = []    

    csv_data.append(["exp. number", "configurations", "loss"])
    for a,b,c in zip(x_plot, x_ticks_configs, y_plot_metric):
        csv_data.append([a, b, c])
        
    with open('plot_data.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csv_data)
    csvFile.close()

    if predictor.predictor_on:
        with open('plot_data_pred.csv' ,'w', newline='') as outFile:
            fileWriter = csv.writer(outFile)
            with open('plot_data.csv','r') as inFile:
                fileReader = csv.reader(inFile)
                for i, row in enumerate(fileReader):
                    if i == 0:
                        row.extend(["intermediate errors", "intermediate errors percentage", "final errors",
                         "final errors percentage", "training addded", "training cancelled", "accurate continuations (refers only to conscious continuations)"])
                        fileWriter.writerow(row)
                    else:
                        row.extend([y_plot_int_errors[i-1], y_plot_int_errors_perc[i-1],
                                     y_plot_f_errors[i-1], y_plot_f_errors_perc[i-1], training_added[i-1], trainings_cancelled[i-1], acc_continuations[i-1]])
                        fileWriter.writerow(row)
        outFile.close()
        inFile.close()
        os.remove("plot_data.csv")
        os.rename("plot_data_pred.csv", "plot_data.csv")
        
        a = acc_cancellations
        
        if predictor.test_predictor_acc:
            with open('plot_data_test_pred_acc.csv' ,'w', newline='') as outFile:
                fileWriter = csv.writer(outFile)
                with open('plot_data.csv','r') as inFile:
                    fileReader = csv.reader(inFile)
                    for i, row in enumerate(fileReader):
                        if i == 0:
                            row.extend(["accurate cancellations"])
                            fileWriter.writerow(row)
                        else:
                            row.extend([acc_cancellations[i-1]])
                            fileWriter.writerow(row)
            outFile.close()
            inFile.close()
            os.remove("plot_data.csv")
            os.rename("plot_data_test_pred_acc.csv", "plot_data.csv")                
        
        
    if direct_on:
        with open('plot_data_dir.csv' ,'w', newline='') as outFile:
            fileWriter = csv.writer(outFile)
            with open('plot_data.csv','r') as inFile:
                fileReader = csv.reader(inFile)
                for i, row in enumerate(fileReader):
                    if i == 0:
                        row.extend(["rectangle volumes", "rectangle diameters"])
                        fileWriter.writerow(row)
                    else:
                        row.extend([rectangles[i-1].size_list, rectangles[i-1].diag_list])
                        fileWriter.writerow(row)
        outFile.close()
        inFile.close()
        os.remove("plot_data.csv")
        os.rename("plot_data_dir.csv", "plot_data.csv")
        
        
def cvalidation_data_to_csv(cvalidation_data):
    try:
        os.remove("cval_data.csv")
    except:
        logger.log("no file 'cval_data' to delete")
    
    csv_data = []    
    csv_data.append(["losses", "i_loss", "f_loss"])
    for a,b,c in zip(cvalidation_data[0], cvalidation_data[1], cvalidation_data[2]):
        csv_data.append([a, b, c])        
        
    with open('cval_data.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csv_data)
    csvFile.close()
    

def access_cvalidation_data():
    ret_data = []
    with open('cval_data.csv','r') as rFile:
        fileReader = csv.reader(rFile, delimiter=',')
        for (i,row) in enumerate(fileReader):
            if i != 0:
                r_row = []
                for x in row:
                    r_row.append(eval(x))
            if i != 0: ret_data.append(r_row)            
    rFile.close()
    return ret_data
            


            

