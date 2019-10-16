import numpy as np
import math
from predictor import Predictor
import logger
import tqdm
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import preprocessing
import math_operations as mo
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


CONVERGENCE_STOP_PERC = 0.1
RANDOM_NORMAL_STDDEV = 0.1
LEARNING_RATE = 0.0001

def encode(x_train, y_train, EMBEDDING_DIM=10, epochs = 5000, loss="Square", predictor=Predictor(False), id2node=[], G=np.nan):
    dim = x_train.shape[0]
    import tensorflow as tf
    # TENSORFLOW MODEL
    x = tf.placeholder(tf.float32, shape=(None, dim))
    W1 = tf.cast(tf.Variable(tf.random_normal([dim, EMBEDDING_DIM], mean=0, stddev=RANDOM_NORMAL_STDDEV)), tf.float32)
    b1 = tf.cast(tf.Variable(tf.random_normal([EMBEDDING_DIM], mean=0, stddev=RANDOM_NORMAL_STDDEV)), tf.float32)
    hidden_representation = tf.add(tf.matmul(x,W1), b1)
    y = tf.placeholder(tf.float32, shape=(None, dim))
    W2 = tf.cast(tf.Variable(tf.random_normal([EMBEDDING_DIM, dim], mean=0, stddev=RANDOM_NORMAL_STDDEV)), tf.float32)
    b2 = tf.cast(tf.Variable(tf.random_normal([dim], mean=0, stddev=RANDOM_NORMAL_STDDEV)), tf.float32)
    prediction = tf.nn.sigmoid(tf.add( tf.matmul(hidden_representation, W2), b2))
    
    
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
        loss_function =  tf.reduce_sum(-tf.multiply(y, tf.log(prediction+1.E-6)) - tf.multiply((1. - y), tf.log(1. - prediction+1.E-6)))
    elif loss=="Cross-Entropy2":
        loss_function = -tf.reduce_sum(y * tf.log(prediction+1.E-6) - (1-y)*tf.log(1-prediction+1.E-6))
    elif loss == "Sigmoid Cross-Entropy":
        loss_function = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels= y, logits=prediction))
    else:
        raise Exception("Invalid loss function")
    train_step = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss_function)

    predictor.clear_losses()         
    prev_loss = 1
    #change_percentage = []
    #sihls = []
    #first_sihl = []
    #first_sihl.append(best_clustering_evaluation(mo.transpose(session.run(W1 + b1))))
    #vecs=[]
    for iteration in tqdm.tqdm(range(epochs), desc="Training embeddings", position=0, leave=True):
        session.run(train_step, feed_dict={x: x_train, y: y_train})
        loss = session.run(loss_function, feed_dict={x: x_train, y: y_train})
        #change_percentage.append((loss-prev_loss)*100/prev_loss)
        #if iteration%50==0:
            #sihls.append(best_clustering_evaluation(mo.transpose(session.run(W1 + b1))))
            #vecs.append(session.run(W1 + b1))
        if math.isnan(loss):
            raise Exception("Loss was None")
        if iteration < epochs//10:
            if iteration%10==0: predictor.register_loss(best_clustering_evaluation(mo.transpose(session.run(W1 + b1))))
        else:
            if abs(loss-prev_loss)*100/prev_loss<CONVERGENCE_STOP_PERC and iteration>epochs//2: break
        prev_loss = loss
        if iteration == epochs//10:
            int_pred = predictor.predict_intermediate()
            int_sihl = best_clustering_evaluation(mo.transpose(session.run(W1 + b1)))
            predict_final_bool = predictor.attempt_final_prediction_bool(int_pred, int_sihl)
            final_pred = predictor.predict_final(predict_final_bool)
            training_completed_bool = predictor.complete_training_bool(final_pred, iteration)
            predictor.add_train_data(training_completed_bool, "intermediate", int_sihl)
            if training_completed_bool.aborted:
                break

    #predictions = session.run(prediction, feed_dict={x: x_train, y: y_train}) 
    final_sihl = best_clustering_evaluation(mo.transpose(session.run(W1 + b1)), id2node, G, False)
    
    vectors = session.run(W1 + b1)
    logger.log("Training ended in " + str(iteration), " iterations, loss is " + str(loss) + ", sihlouette coefficient is " + str(final_sihl))
    
    predictor.collect_results(final_pred, final_sihl, training_completed_bool)
    predictor.add_train_data(training_completed_bool, "final", final_sihl)    
    
    if not training_completed_bool.aborted: return vectors, loss, predictor, training_completed_bool
    return vectors, final_pred, predictor, training_completed_bool

    
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


def cluster_evaluation(X, n_clusters, repetitions=5):
    clusters=KMeans(n_clusters=n_clusters)
    clusters.fit(X)
    labels = clusters.labels_
    sihl_coeff = metrics.silhouette_score(X, labels,metric='euclidean')
    

    return sihl_coeff, labels


def best_clustering_evaluation(X, id2node=[], G=[], visualize_bool=False):
    X = preprocessing.StandardScaler().fit_transform(X)
    sihls_labels = [cluster_evaluation(X, n_clusters) for n_clusters in range(2,11)]
    sihls = [x[0] for x in sihls_labels]
    labels = [x[1] for x in sihls_labels]
    best_sihl = max(sihls)
    best_labels = labels[sihls.index(best_sihl)]
        
    if visualize_bool:
        visualize(G, best_labels)
    
    return best_sihl



def visualize(G, labels):
    color_dict = {0:"r", 1:"b", 2:"gold", 3:"palegreen", 4:"m", 5:"darkgray", 6:"yellow", 7:"g", 8:"pink", 9:"black"}
    colors = [color_dict[i] for i in labels]
    plt.figure(3,figsize=(10,10)) 
    nx.drawing.nx_pylab.draw_networkx(G, nx.spring_layout(G), node_color=colors)
    plt.show()

"""
def best_clustering_evaluation(X, id2node=[], G=[], visualize_bool=False):
    X = preprocessing.StandardScaler().fit_transform(X)
    sihls = [cluster_evaluation(X, n_clusters) for n_clusters in range(2,11)]
    best_sihl = max(sihls)    
    best_n_clusters = sihls.index(best_sihl)+2
    
    db = KMeans().fit(X)
    labels = db.labels_
    for i in range(max(labels)):
        if list(labels).count(i)<=2:
            for l in range(len(labels)):
                if labels[l]==i:
                    labels[l] = -1
    best_n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    if visualize_bool:
        show_only = [node for i, node in enumerate(G.nodes())]
        
    
    #logger.log('Estimated number of clusters: %d' % best_n_clusters_)
    
    return best_sihl
"""