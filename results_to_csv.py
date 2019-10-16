import csv
import os
import logger
import plots
import numpy as np



def export_results_to_csv(rec_results, rec_configs, train_results, train_configs, predictor, direct_on=False, rectangles=[], sorted_results=[]):    
        
    y_plot_loss, y_plot_sihlouette = [abs(i) for i in rec_results[1]], [abs(i) for i in rec_results[0]]
    x_plot = [i for i in range(len(y_plot_loss))]
    x_ticks_configs = [(j, [round(k,3) for k in i[0]], i[1]) for j,i in enumerate(rec_configs)]
          
    create_rec_csv(predictor, direct_on, x_plot, x_ticks_configs, y_plot_loss, y_plot_sihlouette, rectangles) 
    
    y_plot_loss, y_plot_sihlouette = [abs(i) for i in train_results[1]], [abs(i) for i in train_results[0]]
    x_plot = [i for i in range(len(y_plot_loss))]
    x_ticks_configs = [(j, [round(k,3) for k in i[0]], i[1]) for j,i in enumerate(train_configs)]
    
    if predictor.predictor_on:
        training_added, training_added_plot, trainings_cancelled, acc_continuations, acc_cancellations = plots.compute_end_stats(predictor, False)
        y_plot_int_errors, y_plot_int_errors_perc, y_plot_f_errors, y_plot_f_errors_perc  = [abs(i) for i in predictor.int_errors],  [abs(i) for i in predictor.int_errors_perc], [abs(i) for i in predictor.f_errors], [abs(i) for i in predictor.f_errors_perc]
        create_train_csv(predictor, direct_on, x_plot, x_ticks_configs, y_plot_loss, y_plot_sihlouette, y_plot_int_errors, y_plot_int_errors_perc, y_plot_f_errors,
               y_plot_f_errors_perc, training_added, trainings_cancelled, acc_continuations, acc_cancellations, rectangles) 
    else:
        create_train_csv(predictor, direct_on, x_plot, x_ticks_configs, y_plot_loss, y_plot_sihlouette)
        
    create_optimal_path_csv(rectangles)
    
    sorted_loss, sorted_sihl = sorted_results[0], sorted_results[1]
    create_sorted_loss_csv(sorted_loss)
    create_sorted_sihl_csv(sorted_sihl)


    
def create_rec_csv(predictor, direct_on, x_plot, x_ticks_configs, y_plot_loss, y_plot_sihlouette, rectangles):
    
    try:
        os.remove("plot_rec_data.csv")
    except:
        logger.log("no file 'plot_rec_data' to delete")
    
    csv_data = []    

    csv_data.append(["rectangle number", "configurations", "loss", "Sihlouette"])
    for a,b,c,d in zip(x_plot, x_ticks_configs, y_plot_loss, y_plot_sihlouette):
        csv_data.append([a, b, c, d])        
    with open('plot_rec_data.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csv_data)
    csvFile.close()                       
        
    if direct_on:
        with open('plot_rec_data_dir.csv' ,'w', newline='') as outFile:
            fileWriter = csv.writer(outFile)
            with open('plot_rec_data.csv','r') as inFile:
                fileReader = csv.reader(inFile)
                for i, row in enumerate(fileReader):
                    if i == 0:
                        row.extend(["created in x DIRECT iteration", "rectangle volumes", "rectangle diameters", "optimal"])
                        fileWriter.writerow(row)
                    else:
                        row.extend([rectangles[i-1].created_in_iteration, rectangles[i-1].rec_size_list, rectangles[i-1].rec_diag_list, rectangles[i-1].optimal])
                        fileWriter.writerow(row)
        outFile.close()
        inFile.close()
        os.remove("plot_rec_data.csv")
        os.rename("plot_rec_data_dir.csv", "plot_rec_data.csv")
        
      
def create_train_csv(predictor, direct_on, x_plot, x_ticks_configs, y_plot_loss, y_plot_sihlouette, y_plot_int_errors, y_plot_int_errors_perc, y_plot_f_errors,
               y_plot_f_errors_perc, training_added, trainings_cancelled, acc_continuations, acc_cancellations, rectangles):
    
    try:
        os.remove("plot_train_data.csv")
    except:
        logger.log("no file 'plot_train_data' to delete")
    
    csv_data = []    

    csv_data.append(["exp. number", "configurations", "loss", "Sihlouette"])
    for a,b,c,d in zip(x_plot, x_ticks_configs, y_plot_loss, y_plot_sihlouette):
        csv_data.append([a, b, c, d])
        
    with open('plot_train_data.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csv_data)
    csvFile.close()

    if predictor.predictor_on:
        with open('plot_train_data_pred.csv' ,'w', newline='') as outFile:
            fileWriter = csv.writer(outFile)
            with open('plot_train_data.csv','r') as inFile:
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
        os.remove("plot_train_data.csv")
        os.rename("plot_train_data_pred.csv", "plot_train_data.csv")
        
        a = acc_cancellations
        
        if predictor.test_predictor_acc:
            with open('plot_train_data_test_pred_acc.csv' ,'w', newline='') as outFile:
                fileWriter = csv.writer(outFile)
                with open('plot_train_data.csv','r') as inFile:
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
            os.remove("plot_train_data.csv")
            os.rename("plot_train_data_test_pred_acc.csv", "plot_train_data.csv")                
        
        
def create_optimal_path_csv(rectangles):
    rec = rectangles[[x.sihl for x in rectangles].index(max([y.sihl for y in rectangles]))]
    path_loss, path_sihl, configs, iterations = [], [], [], []
    while True:
        path_loss.append(rec.loss)
        path_sihl.append(rec.sihl)
        configs.append(rec.configs)
        iterations.append(rec.created_in_iteration)
        if rec.parent_index is np.nan: break
        rec = rectangles[rec.parent_index]        
    path_loss, path_sihl, configs, iterations = path_loss[::-1], path_sihl[::-1], configs[::-1], iterations[::-1]       

    try:
        os.remove("optimal_path_data.csv")
    except:
        logger.log("no file 'optimal_path_data' to delete")
    
    csv_data = []    

    csv_data.append(["optimal path loss", "optimal path sihlouette", "optimal path configurations", "direct iteration"])
    for a,b,c,d in zip(path_loss, path_sihl, configs, iterations):
        csv_data.append([a, b, c, d])        
    with open('optimal_path_data.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csv_data)
    csvFile.close()
    
    
def create_sorted_loss_csv(sorted_loss):
    try:
        os.remove("sorted_loss.csv")
    except:
        logger.log("no file 'sorted_loss' to delete")        

    csv_data = []

    csv_data.append(["loss", "configuration"])
    for a in sorted_loss:
        csv_data.append([a[0], a[1]])        
    with open('sorted_loss.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csv_data)
    csvFile.close()        

    
def create_sorted_sihl_csv(sorted_sihl):
    try:
        os.remove("sorted_sihl.csv")
    except:
        logger.log("no file 'sorted_sihl' to delete")        

    csv_data = []    

    csv_data.append(["sihlouette", "configuration"])
    for a in sorted_sihl:
        csv_data.append([a[0], a[1]])        
    with open('sorted_sihl.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csv_data)
    csvFile.close()      
           
        
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
            


            

