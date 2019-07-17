import numpy as np
import math
from predictor import predictor
import logger
    

def encode(x_train, y_train, EMBEDDING_DIM=10, epochs = 5000, loss="Square", predictor=predictor(False)):
    
    dim = len(x_train[0])
    import tensorflow as tf
    # TENSORFLOW MODEL
    x = tf.placeholder(tf.float32, shape=(None, dim))
    W1 = tf.Variable(tf.random_normal([dim, EMBEDDING_DIM]))
    b1 = tf.Variable(tf.random_normal([EMBEDDING_DIM]))
    hidden_representation = tf.add(tf.matmul(x,W1), b1)
    y = tf.placeholder(tf.float32, shape=(None, dim))
    W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, dim]))
    b2 = tf.Variable(tf.random_normal([dim]))
    prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_representation, W2), b2))
    
    # TRAIN
    session = tf.Session()
    init = tf.global_variables_initializer()
    session.run(init)
    if loss == "Square":
        loss_function = tf.reduce_mean(tf.square(y - prediction))
    elif loss=="Entropy":
        loss_function = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), reduction_indices=[1]))
    else:
        raise Exception("Invalid loss function")
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss_function)
    
    if predictor.predictor_on and epochs < 100:        
            logger.log("Number of epochs too low to run a prediction model. Not worth it.")
            predictor.predictor_on = False
    predictor.clear_losses()
            
    for iteration in range(epochs):
        session.run(train_step, feed_dict={x: x_train, y: y_train})
        loss = session.run(loss_function, feed_dict={x: x_train, y: y_train})
        if iteration % 500==0:
            logger.log('Iter.', iteration,  'loss: ', loss)
        if abs(loss)<0.0015 or math.isnan(loss):
            break
        if (iteration % 10 == 0) and (iteration <= epochs/10):
            predictor.register_loss(loss)
        if iteration == int(epochs/10) + 10:
            int_pred = predictor.predict_intermediate()
            predict_final_bool = predictor.attempt_final_prediction_bool(int_pred, loss)
            final_pred = predictor.predict_final(predict_final_bool, epochs)
            training_completed_bool = predictor.complete_training_bool(final_pred, iteration)
            predictor.add_train_data(training_completed_bool, "intermediate", loss)
            if training_completed_bool.aborted: break
        
    vectors = session.run(W1 + b1)
    logger.log("Training ended in ", iteration, "iterations. Loss is ", loss)    
    
    predictor.collect_results(final_pred, loss, training_completed_bool)
    predictor.add_train_data(training_completed_bool, "final", loss)    
    
    return vectors, loss, predictor, training_completed_bool
   
    
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
    
#    return metrics.silhouette_score(X, labels)
    
    #X = preprocessing.Normalizer().fit_transform(X, 'l2')
    #colors = ['red', 'blue', 'green', 'cyan', 'yellow', 'magenta', 'orange', 'lime']
    #import matplotlib.pyplot as plt
    #for i in range(len(id2words)):
    #    if not show_only is None and not id2words[i] in show_only:
    #        continue
    #    markeredgecolor = colors[labels[i]] if labels[i]!=-1 else 'black'
    #    plt.plot(X[i][0]*0.5+0.5,X[i][1]*0.5+0.5, 'o', markeredgecolor=markeredgecolor, markerfacecolor=markeredgecolor)
    #    plt.annotate(id2words[i], (X[i][0]*0.5+0.5,X[i][1]*0.5+0.5))
    #plt.show()
    
    #import visualize.visualizer
    #visualize.visualizer.visualize_clusters([[id2words[i] for i in range(len(id2words)) if (show_only is None or id2words[i] in show_only) and labels[i]==cluster] for cluster in range(n_clusters_)])