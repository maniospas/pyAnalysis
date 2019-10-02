import numpy as np
import math
from predictor import Predictor
import logger
import tqdm

CONVERGENCE_STOP_PERC = 0

def encode(x_train, y_train, EMBEDDING_DIM=10, epochs = 5000, loss="Square", predictor=Predictor(False)):
    dim = x_train.shape[0]
    import tensorflow as tf
    # TENSORFLOW MODEL
    x = tf.placeholder(tf.float32, shape=(None, dim))
    W1 = tf.cast(tf.Variable(tf.random_normal([dim, EMBEDDING_DIM])), tf.float32)
    b1 = tf.cast(tf.Variable(tf.random_normal([EMBEDDING_DIM])), tf.float32)
    hidden_representation = tf.add(tf.matmul(x,W1), b1)
    y = tf.placeholder(tf.float32, shape=(None, dim))
    W2 = tf.cast(tf.Variable(tf.random_normal([EMBEDDING_DIM, dim])), tf.float32)
    b2 = tf.cast(tf.Variable(tf.random_normal([dim])), tf.float32)
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
        loss_function = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction+1.E-6), reduction_indices=[1]))
    elif loss=="Cross-Entropy":
        loss_function = -tf.reduce_sum(y * tf.log(prediction+1.E-6)+(1-y)*tf.log(1-prediction+1.E-6))
    else:
        raise Exception("Invalid loss function")
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss_function)

    predictor.clear_losses()         
    prev_loss = 1
    for iteration in tqdm.tqdm(range(epochs), desc="Training embeddings"):
        session.run(train_step, feed_dict={x: x_train, y: y_train})
        loss = session.run(loss_function, feed_dict={x: x_train, y: y_train})
        logger.log("iteration:", iteration, "loss:", loss)
        #logger.log("y:", session.run(y, feed_dict={x: x_train, y: y_train}))
        #logger.log("prediction:", session.run(prediction, feed_dict={x: x_train, y: y_train}))        
        if math.isnan(loss):
            raise Exception("Loss was None")
        if iteration < epochs//10:
            predictor.register_loss(loss)
        else:
            if abs(loss-prev_loss)*100/prev_loss<CONVERGENCE_STOP_PERC:
                break
            prev_loss = loss
        if iteration == epochs//10:
            int_pred = predictor.predict_intermediate()
            predict_final_bool = predictor.attempt_final_prediction_bool(int_pred, loss)
            final_pred = predictor.predict_final(predict_final_bool)
            training_completed_bool = predictor.complete_training_bool(final_pred, iteration)
            predictor.add_train_data(training_completed_bool, "intermediate", loss)
            if training_completed_bool.aborted:
                break
        
    vectors = session.run(W1 + b1)
    logger.log("Training ended in ", iteration, "iterations, loss is ", loss, "previous loss is", prev_loss)
    
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
