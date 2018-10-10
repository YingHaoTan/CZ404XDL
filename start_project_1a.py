#
# Project 1, starter code part a
#
import math
import tensorflow as tf
import numpy as np
import pylab as plt
from tensorflow import initializers as init
from tensorflow import layers
from tensorflow.contrib.layers import l2_regularizer
from tensorflow.losses import get_regularization_loss


# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)


NUM_FEATURES = 36
NUM_CLASSES = 6

learning_rate = 0.01
epochs = 1000
batch_size = 32
num_neurons = 10
seed = 10
np.random.seed(seed)

# read train data
train_input = np.loadtxt('sat_train.txt',delimiter=' ')
trainX, train_Y = train_input[:,:36], train_input[:,-1].astype(int)
trainX = scale(trainX, np.min(trainX, axis=0), np.max(trainX, axis=0))
train_Y[train_Y == 7] = 6

trainY = np.zeros((train_Y.shape[0], NUM_CLASSES))
trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1


# experiment with small datasets
trainX = trainX[:1000]
trainX = (trainX - np.mean(trainX, axis=0)) / np.std(trainX, axis=0)
trainY = trainY[:1000]

n = trainX.shape[0]


# Create the model
x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

# Build the graph for the deep nets
hidden_layer = layers.dense(x, 10, activation=tf.nn.selu, kernel_initializer=init.orthogonal(),
                            bias_initializer=init.zeros(),
                            kernel_regularizer=l2_regularizer(1e-6))
output_layer = layers.dense(x, NUM_CLASSES, kernel_initializer=init.orthogonal(), bias_initializer=init.zeros(),
                            kernel_regularizer=l2_regularizer(1e-6))

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=output_layer)
loss = tf.reduce_mean(cross_entropy) + get_regularization_loss()

# Create the gradient descent optimizer with the given learning rate.
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

correct_prediction = tf.cast(tf.equal(tf.argmax(output_layer, 1), tf.argmax(y_, 1)), tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_acc = []
    for i in range(epochs):
        idx = 0
        indices = np.arange(trainX.shape[0])
        np.random.shuffle(indices)

        trainX, trainY = trainX[indices], trainY[indices]

        while idx < 1000:
            nidx = idx + 32
            train_op.run(feed_dict={x: trainX[idx: nidx], y_: trainY[idx: nidx]})
            idx = nidx
        train_acc.append(accuracy.eval(feed_dict={x: trainX, y_: trainY}))

        if i % 100 == 0:
            print('iter %d: accuracy %g' % (i, train_acc[i]))


# plot learning curves
plt.figure(1)
plt.plot(range(epochs), train_acc)
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Train accuracy')
plt.show()

