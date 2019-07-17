import csv
import os
import logger



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
        
        training_added_plot, trainings_cancelled_sum = predictor.compute_end_stats()
        
        training_added = [((i == 0) or (i == 2)) for i in predictor.stats]
        
        y_plot_int_errors = [abs(i) for i in predictor.intermediate_errors]
        y_plot_int_errors_perc = [abs(i) for i in predictor.intermediate_errors_percentage]
        y_plot_f_errors = [abs(i) for i in predictor.final_errors]
        y_plot_f_errors_perc = [abs(i) for i in predictor.final_errors_percentage]
        y_plot_cancel_perc = [(i/j)*100 for i,j in zip(trainings_cancelled_sum, [i+1 for i in range(len(trainings_cancelled_sum))])]

    create_csv(predictor.predictor_on, direct_on, x_plot, x_ticks_configs, y_plot_metric, y_plot_int_errors, y_plot_int_errors_perc, y_plot_f_errors,
               y_plot_f_errors_perc, y_plot_cancel_perc, training_added, rectangles) 

   # create_csv_dict(predictor.predictor_on, x_plot, x_ticks_configs, y_plot_metric, y_plot_int_errors, y_plot_int_errors_perc,
                                     #y_plot_f_errors, y_plot_f_errors_perc, y_plot_cancel_perc, training_added)

    #create_csv_2(predictor.predictor_on, x_plot, x_ticks_configs, y_plot_metric, y_plot_int_errors, y_plot_int_errors_perc, y_plot_f_errors, y_plot_f_errors_perc, y_plot_cancel_perc, training_added) 

    #create_csv_dict_2(predictor.predictor_on, x_plot, x_ticks_configs, y_plot_metric, y_plot_int_errors, y_plot_int_errors_perc, y_plot_f_errors, y_plot_f_errors_perc, y_plot_cancel_perc, training_added)             

    
def create_csv(predictor_on, direct_on, x_plot, x_ticks_configs, y_plot_metric, y_plot_int_errors=[], y_plot_int_errors_perc=[],
                                     y_plot_f_errors=[], y_plot_f_errors_perc=[], y_plot_cancel_perc=[], training_added=[], rectangles=[]):
    
    try:
        os.remove("plot_data.csv")
    except:
        logger.log("no file 'plot_data' to delete")
    
    csv_data = []    

    csv_data.append(["x plot", "x ticks", "y plot loss"])
    for a,b,c in zip(x_plot, x_ticks_configs, y_plot_metric):
        csv_data.append([a, b, c])
        
    with open('plot_data.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csv_data)
    csvFile.close()

    if predictor_on:
        with open('plot_data_pred.csv' ,'w', newline='') as outFile:
            fileWriter = csv.writer(outFile)
            with open('plot_data.csv','r') as inFile:
                fileReader = csv.reader(inFile)
                for i, row in enumerate(fileReader):
                    if i == 0:
                        row.extend(["y plot intermediate errors", "y plot intermediate errors percentage", "y plot final errors",
                         "y plot final errors percentage", "percentage of cancellations", "training added"])
                        fileWriter.writerow(row)
                    else:
                        row.extend([y_plot_int_errors[i-1], y_plot_int_errors_perc[i-1],
                                     y_plot_f_errors[i-1], y_plot_f_errors_perc[i-1], y_plot_cancel_perc[i-1], training_added[i-1]])
                        fileWriter.writerow(row)
        outFile.close()
        inFile.close()
        os.remove("plot_data.csv")
        os.rename("plot_data_pred.csv", "plot_data.csv")
        
    if direct_on:
        with open('plot_data_dir.csv' ,'w', newline='') as outFile:
            fileWriter = csv.writer(outFile)
            with open('plot_data.csv','r') as inFile:
                fileReader = csv.reader(inFile)
                for i, row in enumerate(fileReader):
                    if i == 0:
                        row.extend(["rectangle volume"])
                        fileWriter.writerow(row)
                    else:
                        row.extend([rectangles[i-1].size])
                        fileWriter.writerow(row)
        outFile.close()
        inFile.close()
        os.remove("plot_data.csv")
        os.rename("plot_data_dir.csv", "plot_data.csv")
            
    

    
    
"""def create_csv_dict(predictor_on, x_plot, x_ticks_configs, y_plot_metric, y_plot_int_errors, y_plot_int_errors_perc,
                                     y_plot_f_errors, y_plot_f_errors_perc, y_plot_cancel_perc, training_added):
    
    try:
        os.remove("plot_data_dict.csv")
    except:
        logger.log("no file 'plot_data_dict' to delete")
    
    csv_data_dict= []
    if predictor_on:
        fieldnames = ["x plot", "x ticks", "y plot loss", "y plot intermediate errors", "y plot intermediate errors percentage", "y plot final errors",
                         "y plot final errors percentage", "percentage of cancellations", "training added"]        
        for a, b, c, d, e, f, g, h, i in zip(x_plot, x_ticks_configs, y_plot_metric, y_plot_int_errors, y_plot_int_errors_perc, y_plot_f_errors, y_plot_f_errors_perc, y_plot_cancel_perc, training_added):
            csv_data_dict.append({"x plot": a, "x ticks": b, "y plot loss": c, "y plot intermediate errors": d, "y plot intermediate errors percentage": e,
                                  "y plot final errors": f, "y plot final errors percentage": g, "percentage of cancellations": h, "training added": i})
    else:
        fieldnames = ["x plot", "x ticks", "y plot loss"]
        for a, b, in zip(x_plot, x_ticks_configs, y_plot_metric):
            csv_data_dict.append({"x plot": a, "x ticks": b, "y plot loss": c})
            
    with open('plot_data_dict.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in csv_data_dict:
            writer.writerow(row)

    csvfile.close()"""
    
    
"""def create_csv_2(predictor_on, x_plot, x_ticks_configs, y_plot_metric, y_plot_int_errors, y_plot_int_errors_perc,
                                     y_plot_f_errors, y_plot_f_errors_perc, y_plot_cancel_perc, training_added):
    
    try:
        os.remove("plot_data_excel.csv")
    except:
        logger.log("no file 'plot_data_excel' to delete")
    
    csv_data_excel = []    
    if predictor_on:
        csv_data_excel.append(["x plot", "x ticks", "y plot loss", "y plot intermediate errors", "y plot intermediate errors percentage", "y plot final errors",
                         "y plot final errors percentage", "percentage of cancellations", "training added"])
        for a, b, c, d, e, f, g, h, i in zip(x_plot, x_ticks_configs, y_plot_metric, y_plot_int_errors, y_plot_int_errors_perc,
                                     y_plot_f_errors, y_plot_f_errors_perc, y_plot_cancel_perc, training_added):
            csv_data_excel.append([a, b, c, d, e, f, g, h, i])
    else:
        csv_data_excel.append("x plot", "x ticks", "y plot loss")
        for a,b,c in zip(x_plot, x_ticks_configs, y_plot_metric):
            csv_data_excel.append(a, b, c)
    
    with open('plot_data_excel.csv', 'w') as csvFile:
        writer = csv.writer(csvFile, dialect='excel')
        writer.writerows(csv_data_excel)
        
    csvFile.close()"""
    
    
"""def create_csv_dict_2(predictor_on, x_plot, x_ticks_configs, y_plot_metric, y_plot_int_errors, y_plot_int_errors_perc,
                                     y_plot_f_errors, y_plot_f_errors_perc, y_plot_cancel_perc, training_added):
    
    try:
        os.remove("plot_data_dict_excel.csv")
    except:
        logger.log("no file 'plot_data_dict_excel' to delete")
    
    csv_data_dict_excel= []
    if predictor_on:
        fieldnames = ["x plot", "x ticks", "y plot loss", "y plot intermediate errors", "y plot intermediate errors percentage", "y plot final errors",
                         "y plot final errors percentage", "percentage of cancellations", "training added"]        
        for a, b, c, d, e, f, g, h, i in zip(x_plot, x_ticks_configs, y_plot_metric, y_plot_int_errors, y_plot_int_errors_perc, y_plot_f_errors, y_plot_f_errors_perc, y_plot_cancel_perc, training_added):
            csv_data_dict_excel.append({"x plot": a, "x ticks": b, "y plot loss": c, "y plot intermediate errors": d, "y plot intermediate errors percentage": e,
                                  "y plot final errors": f, "y plot final errors percentage": g, "percentage of cancellations": h, "training added": i})
    else:
        fieldnames = ["x plot", "x ticks", "y plot loss"]
        for a, b, in zip(x_plot, x_ticks_configs, y_plot_metric):
            csv_data_dict_excel.append({"x plot": a, "x ticks": b, "y plot loss": c})
            
    with open('plot_data_dict_excel.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, dialect="excel")

        writer.writeheader()
        for row in csv_data_dict_excel:
            writer.writerow(row)

    csvfile.close()"""



