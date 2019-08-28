from sklearn.svm import SVR
import numpy as np
import logger
import math
import topic_analysis as topic
import sklearn.metrics


ST_DEV_MUL = 1
INT_PRED_LIM = 10
COMPLETE_FIRST_TRAININGS = 10

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
        self.params = svr_params      # 3 means the training was correctly continued, 4 means the training was wrongly continued
        self.test_predictor_acc = test_predictor_acc
        self.acc_cancellations = []
        self.acc_continuations = []
        self.cvalidation_data = [[] for i in range(3)]
        self.losses_std = []
        logger.log("Predictor initiated. Prediction model: SVR. Parameters:", self.params)

    
    def register_loss(self, loss):        
        if self.predictor_on:
            self.losses.append(math.log(loss))

        
    def predict_intermediate(self):        
        if self.predictor_on:
            if len(self.X_train) >= 1:
                return SVR(kernel=self.params[0], C=self.params[1], gamma=self.params[2]).fit(np.asarray(self.X_train), np.asarray(self.Y_train_int)).predict(np.asarray([self.losses]))    
            else:
                return math.e #some big value
        return None

    
    def attempt_final_prediction_bool(self, pred, loss):        
        if self.predictor_on:
            if self.test_predictor_acc:
                self.cvalidation_data[0].append(self.losses)
                self.cvalidation_data[1].append(math.log(loss))
            self.int_errors.append(math.exp(pred) - loss)
            self.int_errors_perc.append((math.exp(pred) - loss)/loss*100)
            if (abs(math.exp(pred) - loss)/loss)*100 > INT_PRED_LIM:
                logger.log("intermediate prediction is inacurrate by", ((math.exp(pred) - loss)/loss)*100, "%")
                return False
            else:
                logger.log("intermediate prediction is accurate enough. The difference is", ((math.exp(pred) - loss)/loss)*100, "%")
                return True
        return False

        
    def predict_final(self, predict_final_bool):        
        if self.predictor_on:
            if predict_final_bool:
                return SVR(kernel=self.params[0], C=self.params[1], gamma=self.params[2]).fit(np.asarray(self.X_train), np.asarray(self.Y_train_f)).predict(np.asarray([self.losses]))    
            return None 
        return None

        
    def complete_training_bool(self, pred_final, iteration):        
        if pred_final is None:  # meaning either predictor is turned off, or the earlier prediction wasnt accurate enough, so we didnt attempt final one
            return training_continued()
        else:
            if self.loss_is_good(math.exp(pred_final), True) or len(self.loss_list) < COMPLETE_FIRST_TRAININGS:
                if len(self.loss_list) > COMPLETE_FIRST_TRAININGS: logger.log("Prediction of final loss in iteration number " + str(iteration) + " is good. " + "Training continues...")
                return training_continued()
            else:
                logger.log("Prediction of final loss in iteration number", iteration, "is too high. Training aborted...")
                if self.test_predictor_acc: return training_continued() # if we choose this mode, we want to check and compare the final prediction everytime, so that we have more comparisons
                return training_aborted()
            
            
    def add_train_data(self, complete_training_bool, intermediate_or_final, loss):        
        if self.predictor_on:
            if not complete_training_bool.aborted:
                if intermediate_or_final == "intermediate":
                    self.X_train.append(self.losses)
                    self.Y_train_int.append(math.log(loss))
                elif intermediate_or_final == "final":
                    self.Y_train_f.append(math.log(loss))

                    
    def collect_results(self, pred_final, loss, complete_training_bool):
        if self.test_predictor_acc:
            self.cvalidation_data[2].append(math.log(loss))
        if pred_final is None or len(self.loss_list) < COMPLETE_FIRST_TRAININGS:
            self.stats.append(0) # the intermediate prediction was inaccurate or the predictor is turned not active
            #logger.log("appended 0")
            self.f_errors.append(np.nan)
            self.f_errors_perc.append(np.nan)
            self.loss_list.append(loss)
        else:
            if complete_training_bool.aborted:
                self.stats.append(1) # the intermediate prediction was accurate but the final prediction was bad, so the training was aborted
                #logger.log("appended 1")
                self.f_errors.append(np.nan)
                self.f_errors_perc.append(np.nan)
                self.loss_list.append(math.exp(pred_final))
            else:
                if not self.loss_is_good(math.exp(pred_final), False) and self.test_predictor_acc: #meaning the training was aborted but we want to continue for testing stats purposes
                    self.stats.append(1) # even if we continue with the training because we want to test accuracy of final prediction every time, we want to keep our stats right   
                    #logger.log("appended 1")
                    if loss < min(self.loss_list):
                        self.stats[-1] = 2   # the training was wrongly cancelled
                        #logger.log("appended 2")
                else:
                    self.stats.append(3) # the intermediate prediction was accurate and the final prediction was good, so the training continued
                    #logger.log("appended 3")
                    if not self.loss_is_good(loss, False):
                        self.stats[-1] = 4  # the training wrongly continued                 
                        #logger.log("appended 4")
                logger.log("The final prediction error percentage is:", (math.exp(pred_final) - loss)/loss*100, "%:", math.exp(pred_final) - loss)
                self.f_errors_perc.append((math.exp(pred_final) - loss)/loss*100)
                self.f_errors.append(math.exp(pred_final) - loss)
                self.loss_list.append(loss)
        #logger.log(self.loss_list, np.mean(self.loss_list), np.std(self.loss_list))        
        
        
    def clear_losses(self):        
        if self.predictor_on:
            self.losses = []
    
    def loss_is_good(self, loss, print_res):
        if print_res:
            if self.loss_is_good(loss, False):
                logger.log(loss, "<", min(self.loss_list), "+",  np.mean([abs(x) for x in self.int_errors[1:]]), "+", ST_DEV_MUL*np.std(self.int_errors[1:]), "=", min(self.loss_list) + np.mean([abs(x) for x in self.int_errors[1:]]) + ST_DEV_MUL*np.std(self.int_errors[1:]))
            else:
                logger.log(loss, ">=", min(self.loss_list), "+",  np.mean([abs(x) for x in self.int_errors[1:]]), "+", ST_DEV_MUL*np.std(self.int_errors[1:]), "=", min(self.loss_list) + np.mean([abs(x) for x in self.int_errors[1:]]) + ST_DEV_MUL*np.std(self.int_errors[1:]))
        return loss < min(self.loss_list) + np.mean([abs(x) for x in self.int_errors[1:]]) + ST_DEV_MUL*np.std(self.int_errors[1:])
 
    
class training_continued:
    def __init__(self):
        self.aborted = False
    
    def return_metrics(self, loss, vectors, id2node, function_names, G):
        sihlouette = topic.cluster(vectors, id2node, function_names)
        vectors = {id2node[u]: vectors[u] for u in range(len(vectors))}
        auc = sklearn.metrics.roc_auc_score([1 if G.has_edge(u, v) or G.has_edge(v, u) else 0 for u in G.nodes() for v in G.nodes() if v!=u], [topic.similarity(vectors[u], vectors[v]) for u in G.nodes() for v in G.nodes() if v!=u])
        return sihlouette, auc, loss           
        
    
class training_aborted:
    def __init__(self):
        self.aborted = True

    def return_metrics(self, loss, vectors, id2node, function_names, G):
        return -math.inf, -math.inf, math.inf

    