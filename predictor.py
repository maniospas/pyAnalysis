from sklearn.svm import SVR
import numpy as np
import logger
import math
import topic_analysis as topic
import sklearn.metrics
import math_operations as math_ops
import time
from sklearn.preprocessing import StandardScaler
import copy

MUL = 1.5
INT_PRED_LIM = 5
COMPLETE_FIRST_TRAININGS = 30
AUC_NEG_SAMPLES_PERC = 2.5

class Predictor:
    def __init__(self, predictor_on, svr_params=None, test_predictor_acc=False):
        self.predictor_on = predictor_on
        self.losses = []
        self.loss_list = []
        self.X_train = []
        self.Y_train_int = []
        self.Y_train_f = []
        self.stats = []                   # stats contains one number (0,1,2,3,4) for each configuration we run. they are explained below
        self.int_errors = []              # appending 0 means either predictor is turned off, or it is turned on and the intermediate prediction wasnt accurate enough, so we didnt attempt final one
        self.int_errors_perc = []         # appending 1 or 2 means the intermediate prediction was accurate but the final prediction was bad, so the training was aborted 
        self.f_errors = []                # 1 means the training was correctly cancelled, 2 means the training was wrongly cancelled (we have this distinction only if test_predictor_acc mode is turned on. Else, we append just 1 for every cancellation)
        self.f_errors_perc = []           # appending 3 or 4 means the intermediate prediction was accurate and the final prediction was good, so the training continued
        self.params = svr_params          # 3 means the training was correctly continued, 4 means the training was wrongly continued
        self.test_predictor_acc = test_predictor_acc
        self.acc_cancellations = []
        self.acc_continuations = []
        self.cvalidation_data = [[] for i in range(3)]
        logger.log("Predictor initiated. Prediction model: SVR. Parameters:", self.params)

    
    def register_loss(self, loss):        
        if self.predictor_on:
            self.losses.append(loss)

        
    def predict_intermediate(self):        
        if self.predictor_on:
            if len(self.X_train) >= 1:
                scaler = StandardScaler()
                x = copy.deepcopy(self.X_train)
                x.append(self.losses)
                x_scaled = scaler.fit_transform(x)
                return SVR(kernel=self.params[0], C=self.params[1], gamma=self.params[2], epsilon=self.params[3]).fit(np.asarray(x_scaled[:-1]), np.asarray(self.Y_train_int)).predict(np.asarray([x_scaled[-1]]))    
            else:
                return math.e #some random value
        return None

    
    def attempt_final_prediction_bool(self, pred, loss):        
        if self.predictor_on:
            if self.test_predictor_acc:
                self.cvalidation_data[0].append(self.losses)
                self.cvalidation_data[1].append(loss)
            self.int_errors.append(pred - loss)
            self.int_errors_perc.append((pred - loss)/loss*100)
            if len(self.X_train) >= COMPLETE_FIRST_TRAININGS:
                if (abs(pred - loss)/loss)*100 > INT_PRED_LIM:
                    logger.log("intermediate prediction is inacurrate by", ((pred - loss)/loss)*100, "%")
                    return False
                else:
                    logger.log("intermediate prediction is accurate enough. The difference is", ((pred - loss)/loss)*100, "%")
                    return True
            else: 
                return False
        return False

        
    def predict_final(self, predict_final_bool):        
        if self.predictor_on:
            if predict_final_bool:
                scaler = StandardScaler()
                x = copy.deepcopy(self.X_train)
                x.append(self.losses)
                x_scaled = scaler.fit_transform(x)
                return SVR(kernel=self.params[0], C=self.params[1], gamma=self.params[2], epsilon=self.params[3]).fit(np.asarray(x_scaled[:-1]), np.asarray(self.Y_train_f)).predict(np.asarray([x_scaled[-1]]))    
            return None 
        return None

        
    def complete_training_bool(self, pred_final, iteration):        
        if pred_final is None:  # meaning either predictor is turned off, or the earlier prediction wasnt accurate enough, so we didnt attempt final one
            return training_continued()
        else:
            if self.loss_is_good(pred_final, True) or len(self.loss_list) < COMPLETE_FIRST_TRAININGS:
                if len(self.loss_list) > COMPLETE_FIRST_TRAININGS: logger.log("Prediction of final loss in iteration number " + str(iteration) + " is good. " + "Training continues...")
                return training_continued()
            else:
                logger.log("Prediction of final sihlouette value in iteration number", iteration, "is too low:", pred_final,". Training aborted...")
                if self.test_predictor_acc: return training_continued() # if we choose this mode, we want to check and compare the final prediction everytime, so that we have more comparisons
                return training_aborted()
            
            
    def add_train_data(self, complete_training_bool, intermediate_or_final, loss):        
        if self.predictor_on:
            if not complete_training_bool.aborted:
                if intermediate_or_final == "intermediate":
                    self.X_train.append(self.losses)
                    self.Y_train_int.append(loss)
                elif intermediate_or_final == "final":
                    self.Y_train_f.append(loss)

                    
    def collect_results(self, pred_final, loss, complete_training_bool):
        if self.test_predictor_acc:
            self.cvalidation_data[2].append(loss)
        if pred_final is None or len(self.loss_list) < COMPLETE_FIRST_TRAININGS:
            self.stats.append(0) # the intermediate prediction was inaccurate or the predictor is turned not active
            self.f_errors.append(np.nan)
            self.f_errors_perc.append(np.nan)
            self.loss_list.append(loss)
        else:
            if complete_training_bool.aborted:
                self.stats.append(1) # the intermediate prediction was accurate but the final prediction was bad, so the training was aborted
                self.f_errors.append(np.nan)
                self.f_errors_perc.append(np.nan)
                self.loss_list.append(pred_final)
            else:
                if not self.loss_is_good(pred_final, False) and self.test_predictor_acc: #meaning the training was aborted but we want to continue for testing stats purposes
                    self.stats.append(1) # even if we continue with the training because we want to test accuracy of final prediction every time, we want to keep our stats right   
                    if loss < min(self.loss_list):
                        self.stats[-1] = 2   # the training was wrongly cancelled
                else:
                    self.stats.append(3) # the intermediate prediction was accurate and the final prediction was good, so the training continued
                    if not self.loss_is_good(loss, False):
                        self.stats[-1] = 4  # the training wrongly continued                 
                logger.log("The final prediction error percentage is:", (pred_final - loss)/loss*100, "%:", pred_final - loss)
                self.f_errors_perc.append((pred_final - loss)/loss*100)
                self.f_errors.append(pred_final - loss)
                self.loss_list.append(loss)
        
        
    def clear_losses(self):        
        if self.predictor_on:
            self.losses = []
    
    def loss_is_good(self, loss, print_res):
        if print_res:
            if self.loss_is_good(loss, False):
                logger.log(loss, ">", max(self.loss_list), "-", np.std(self.loss_list), "-" , MUL*np.mean([abs(x) for x in self.int_errors[COMPLETE_FIRST_TRAININGS:]]), "-", MUL*np.std([abs(x) for x in self.int_errors[COMPLETE_FIRST_TRAININGS:]]), "=", max(self.loss_list) - np.std(self.loss_list) - MUL*np.mean([abs(x) for x in self.int_errors[COMPLETE_FIRST_TRAININGS-1:]]) - MUL*np.std([abs(x) for x in self.int_errors[COMPLETE_FIRST_TRAININGS:]]))
            else:
                logger.log(loss, "<", max(self.loss_list), "-", np.std(self.loss_list), "-" , MUL*np.mean([abs(x) for x in self.int_errors[COMPLETE_FIRST_TRAININGS:]]), "-", MUL*np.std(self.int_errors[COMPLETE_FIRST_TRAININGS:]), "=", max(self.loss_list) - np.std(self.loss_list) - MUL*np.mean([abs(x) for x in self.int_errors[COMPLETE_FIRST_TRAININGS-1:]]) - MUL*np.std([abs(x) for x in self.int_errors[COMPLETE_FIRST_TRAININGS:]]))
        return loss > max(self.loss_list) - np.std(self.loss_list) - MUL*np.mean([abs(x) for x in self.int_errors[COMPLETE_FIRST_TRAININGS-1:]]) - MUL*np.std([abs(x) for x in self.int_errors[COMPLETE_FIRST_TRAININGS:]])
 
    
class training_continued:
    def __init__(self):
        self.aborted = False
    
    def return_metrics(self, loss, vectors, id2node, predictor):
        sihlouette = predictor.Y_train_f[-1]
        vectors = {id2node[u]: vectors[u] for u in range(len(vectors))}
        return sihlouette, loss           
        
    
class training_aborted:
    def __init__(self):
        self.aborted = True

    def return_metrics(self, loss, vectors, id2node, predictor):
        return -math.inf, math.inf


