import numpy as np
import math
from predictor import Predictor
import logger
    
EPOCH_LIM = 1000
CONVERGENCE_STOP_PERC = 0.07

def encode(x_train, y_train, EMBEDDING_DIM=10, epochs = 5000, loss="Square", predictor=Predictor(False)): 
    dim = len(x_train[0])
    import tensorflow as tf
    # TENSORFLOW MODEL
    x = tf.placeholder(tf.float64, shape=(None, dim))
    W1 = tf.cast(tf.Variable(tf.random_normal([dim, EMBEDDING_DIM])), tf.float64)
    b1 = tf.cast(tf.Variable(tf.random_normal([EMBEDDING_DIM])), tf.float64)
    hidden_representation = tf.add(tf.matmul(x,W1), b1)
    y = tf.placeholder(tf.float64, shape=(None, dim))
    W2 = tf.cast(tf.Variable(tf.random_normal([EMBEDDING_DIM, dim])), tf.float64)
    b2 = tf.cast(tf.Variable(tf.random_normal([dim])), tf.float64)
    prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_representation, W2), b2))
    
    # TRAIN
    session = tf.Session()
    init = tf.global_variables_initializer()
    session.run(init)
    if loss == "Square":
        loss_function = tf.reduce_mean(tf.square(y - prediction))
    elif loss == "Absolute":
        loss_function = tf.reduce_mean(tf.abs(y - prediction))
    elif loss == "Pseudo-Huber":
        delta = tf.constant(0.24)
        loss_function = tf.reduce_mean(tf.multiply(tf.square(delta),tf.sqrt(1. + tf.square((y - prediction) / delta)) - 1. ))
    elif loss=="Entropy":
        loss_function = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), reduction_indices=[1]))
    elif loss=="Entropy2":
        loss_function = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction)
    else:
        raise Exception("Invalid loss function")
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss_function)

    predictor.clear_losses()         
    prev_loss, iteration = 1, 0
    while iteration <= epochs:
        session.run(train_step, feed_dict={x: x_train, y: y_train})
        loss = session.run(loss_function, feed_dict={x: x_train, y: y_train})
        if math.isnan(loss):
            loss = prev_loss
            break
        if iteration % 500==0:
            logger.log('Iter.', iteration,  'loss: ', loss)
        if (iteration % 100  == 0) and (iteration <= EPOCH_LIM):
            predictor.register_loss(loss)
        if (iteration % 10 == 0):
            if (abs(loss-prev_loss)*100/prev_loss<CONVERGENCE_STOP_PERC) and (iteration>(EPOCH_LIM+101)): break
            prev_loss = loss
        if iteration == EPOCH_LIM + 100:
            int_pred = predictor.predict_intermediate()
            predict_final_bool = predictor.attempt_final_prediction_bool(int_pred, loss)
            final_pred = predictor.predict_final(predict_final_bool)
            training_completed_bool = predictor.complete_training_bool(final_pred, iteration)
            predictor.add_train_data(training_completed_bool, "intermediate", loss)
            if training_completed_bool.aborted: break
        iteration = iteration+1
        
    vectors = session.run(W1 + b1)
    logger.log("Training ended in ", iteration, "iterations. Loss is ", loss)    
    
    predictor.collect_results(final_pred, loss, training_completed_bool)
    predictor.add_train_data(training_completed_bool, "final", loss)    
    
    if not training_completed_bool.aborted: return vectors, loss, predictor, training_completed_bool
    return vectors,math.exp(final_pred), predictor, training_completed_bool

    
def one_hot(index, size):
    temp = np.zeros(size)
    temp[index] = 1
    return temp


def features(sentences, WINDOW_SIZE=2): 
    # CREATE DICT
    words = set([word for sentence in sentences for word in sentence])
    word2int = {word: i for i, word in enumerate(words)}
    int2word = {word2int[word]: word for word in words}
    
    # CREATE DATA
    x_train = []
    y_train = []
    for sentence in sentences:
        for word_index, word in enumerate(sentence):
            for nb_word in sentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1] : 
                if nb_word != word:
                    x_train.append(one_hot(word2int[word], len(words)))
                    y_train.append(one_hot(word2int[nb_word], len(words)))
    return x_train, y_train, word2int, int2word


def similarity(v1, v2):
    return sum(v1[i]*v2[i] for i in range(len(v1)))#/np.sqrt(sum(v1[i]*v1[i] for i in range(len(v1))))/np.sqrt(sum(v2[i]*v2[i] for i in range(len(v1))))


def cluster(X, id2words, show_only):
    from sklearn.cluster import KMeans
    from sklearn import metrics
    from sklearn import preprocessing
    X = preprocessing.StandardScaler().fit_transform(X)
    
    db = KMeans().fit(X)
    labels = db.labels_
    for i in range(max(labels)):
        if list(labels).count(i)<=2:
            for l in range(len(labels)):
                if labels[l]==i:
                    labels[l] = -1
    
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    
    #logger.log('Estimated number of clusters: %d' % n_clusters_)
    #logger.log("Silhouette Coefficient: %0.3f"  % metrics.silhouette_score(X, labels))
    
    return metrics.silhouette_score(X, labels)
