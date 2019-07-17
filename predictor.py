from sklearn.svm import SVR
import training
import numpy as np
import logger
import math
from statsmodels.tsa.arima_model import ARMA, ARMAResults, ARIMA



class predictor:
    def __init__(self, predictor_on, model_type="svr", svr_params=None, arima_params=None, arma_params=[1, 1], test_predictor_acc=False):
        self.predictor_on = predictor_on
        self.model_type = model_type
        self.losses = []
        self.loss_list = []
        self.best_losses = []
        self.best_losses.extend([1,1]) # load 2 big values 
        self.X_train = []
        self.Y_train_int = []
        self.Y_train_f = []
        self.stats = []  # stats contains one number (0,1,2) for each configuration we run. they are explained below
        self.int_errors = []
        self.int_errors_perc = []
        self.f_errors = []
        self.f_errors_perc = []
        self.svr_params = svr_params
        self.arima_params = arima_params
        self.arma_params = arma_params
        self.test_predictor_acc = test_predictor_acc
        self.completed_runs = 0
        if self.model_type == "svr":
            self.params = self.svr_params
        elif self.model_type == "ARIMA":
            self.params = self.arima_params
        elif self.model_type == "ARMA":
            self.params = self.arma_params
        logger.log("Predictor initiated. Type:", self.model_type, ". Parameters:", self.params)

    
    def register_loss(self, loss):
        
        if self.predictor_on:
            self.losses.append(math.log(loss))
        else: 
            pass

        
    def predict_intermediate(self):
        
        if self.predictor_on:
            if self.model_type == "svr":
                if len(self.X_train) >= 1:
                    return SVR(kernel=self.params[0], C=self.params[1], gamma=self.params[2]).fit(np.asarray(self.X_train), np.asarray(self.Y_train_int)).predict(np.asarray([self.losses]))    
                else:
                    return math.e #some big value
            elif self.model_type == "ARIMA":
                return ARIMA(self.losses, order=(self.params[0], self.params[1], self.params[2])).fit(disp=0).predict(len(self.losses), len(self.losses))[0]
            elif self.model_type == "ARMA":
                return ARMA(self.losses, order=(self.params[0], self.params[1])).fit(disp=0).predict(len(self.losses), len(self.losses))[0]                
        else:
            return None

    
    def attempt_final_prediction_bool(self, pred, loss):
        
        if self.predictor_on:
            self.int_errors.append(math.exp(pred) - loss)
            self.int_errors_perc.append((math.exp(pred) - loss)/loss*100)
            if abs(math.exp(pred) - loss) > 0.001:
                logger.log("intermediate prediction is inacurrate by", math.exp(pred) - loss)
                return False
            else:
                logger.log("intermediate prediction is accurate enough. The difference is", math.exp(pred) - loss)
                return True
        else:
            return False

        
    def predict_final(self, predict_or_not, epochs):
        
        if self.predictor_on:
            if predict_or_not:
                if self.model_type == "svr":
                    return SVR(kernel=self.params[0], C=self.params[1], gamma=self.params[2]).fit(np.asarray(self.X_train), np.asarray(self.Y_train_f)).predict(np.asarray([self.losses]))    
                elif self.model_type == "ARIMA":
                    return ARIMA(self.losses, order=(self.params[0], self.params[1], self.params[2])).fit(disp=0).predict(len(self.losses), int(epochs/10))[-1]
                elif self.model_type == "ARMA":
                    return ARMA(self.losses, order=(self.params[0], self.params[1])).fit(disp=0).predict(len(self.losses), int(epochs/10))[-1]

            else:
                return None
        else: 
            return None

        
    def complete_training_bool(self, pred_final, iteration):
        
        if pred_final is None:  # meaning either predictor is turned off, or the earlier prediction wasnt accurate enough, so we didnt attempt final one
            return training.training_continued()
        else:
            if math.exp(pred_final) >= self.best_losses[-1]:
                logger.log("Prediction of final loss in iteration number", iteration, "is too high. Training aborted...")
                if self.test_predictor_acc: return training.training_continued() # if we choose this mode, we want to check and compare the final prediction everytime, so that we have more comparisons
                return training.training_aborted()
            else:
                logger.log("Prediction of final loss in iteration number", iteration, "is in the best", len(self.best_losses) ,"losses enquentered so far. Training continues...")
                return training.training_continued()

            
    def add_train_data(self, run_or_not, intermediate_or_final, loss):
        
        if self.predictor_on:
            if not run_or_not.aborted:
                if intermediate_or_final == "intermediate":
                    self.X_train.append(self.losses)
                    self.Y_train_int.append(math.log(loss))
                elif intermediate_or_final == "final":
                    self.Y_train_f.append(math.log(loss))

                    
    def collect_results(self, pred_final, loss, run_or_not):

        if pred_final is None:
            self.stats.append(0) # the intermediate prediction was inaccurate
            self.f_errors.append(np.nan)
            self.f_errors_perc.append(np.nan)
            self.loss_list.append(loss)
        else:
            if run_or_not.aborted:
                self.stats.append(1) # the intermediate prediction was accurate but the final prediction was bad, so the training was aborted
                self.f_errors.append(np.nan)
                self.f_errors_perc.append(np.nan)
                self.loss_list.append(math.exp(pred_final))
            else:
                if math.exp(pred_final) >= self.best_losses[-1] and self.test_predictor_acc:
                    self.stats.append(1) # even if we continue with the training because we want to test accuracy of final prediction every time, we want to keep our stats right   
                else:
                    self.stats.append(2) # the intermediate prediction was accurate and the final prediction was good, so the training continued
                logger.log("The final prediction error percentage is:", (math.exp(pred_final) - loss)/loss*100, "%:", math.exp(pred_final) - loss)
                self.f_errors_perc.append((math.exp(pred_final) - loss)/loss*100)
                self.f_errors.append(math.exp(pred_final) - loss)
                self.loss_list.append(loss)

        
    def clear_losses(self):
        
        if self.predictor_on:
            self.losses = []
        else:
            pass

        
    def loss_in_best_losses(self, run_or_not, loss):
        
        if self.predictor_on:        
            if not run_or_not.aborted:
                s = np.std(self.loss_list)
                measure = self.best_losses[0] + 2*s
                if loss < measure:
                    import bisect
                    bisect.insort(self.best_losses, loss)
                for i in self.best_losses:
                    if i > measure: self.best_losses.remove(i)
            logger.log("best losses:", self.best_losses)
            logger.log("loss list:", self.loss_list)
                    
    
    def print_percentage_of_complete_runs(self, full_run_counter, total_experiments, iterations):
        
        if self.predictor_on:
            logger.log("Out of", total_experiments*iterations, "training procedures,", full_run_counter,
                            "of them were completed. That is", full_run_counter*100/(total_experiments*iterations), "%")
        else:
            pass

        
    def return_int_errors_mean(self):
        
        if self.predictor_on:
            return sum([abs(x) for x in self.int_errors])/len(self.int_errors)
        else: 
            return None

        
    def return_f_errors_mean(self):
        
        if self.predictor_on:
            f_errors = [x for x in self.f_errors if not np.isnan(x)]
            return sum([abs(x) for x in f_errors])/len(f_errors) if f_errors is not None else math.inf
        else:
            return None


    def compute_end_stats(self):  
        
        training_added = [((i == 0) or (i == 2)) for i in self.stats]
        training_added_plot = []
        for i,j in enumerate(training_added):
            if j == True:
                training_added_plot.append(i)
                self.completed_runs += 1
                    
        trainings_cancelled = [(i == 1) for i in self.stats]
        trainings_cancelled_sum, trainings_cancelled_plot, count = [], [], 0
        for i,j in enumerate(trainings_cancelled):
            if j == True:
                count += 1
                trainings_cancelled_plot.append(i)
            trainings_cancelled_sum.append(count)  
            
        return training_added_plot, trainings_cancelled_sum
    