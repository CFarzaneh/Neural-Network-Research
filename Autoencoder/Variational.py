import tensorflow as tf
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
from IPython import embed
import sys

def genData(numPts):
# These are the basis vectors. They are unit vectors
# so we can easily control the magnitude
    u1 = np.array((1,1))/np.sqrt(2)
    u2 = np.array((-1,1))/np.sqrt(2)
    basis = [u1, u2]

    magnitude = 3

# labels is an array with 'datasetSize' 0's and 1's
# chosen from a binomial distribution with equal probability
    labels = np.random.binomial(1, 0.5, numPts) #either 0 or 1
# scale is an array with 'datasetSize' floats between
# 0 and magnitude. These are used for scaling the vectors
    scales = np.random.uniform(-magnitude, magnitude, numPts) #Between -3 to 3 vector

# construct the dataset by scaling each of the randomly selected
# basis vectors. We're only using list comprehension here. There is
# probablyi a slicker way of doing this in the future
    dataset = np.array([scale*basis[index] for (scale,index) in
        zip(scales,labels)], dtype='float32')

    return (dataset, labels)

n_visible = 2
n_hidden = 1
datasetSize = 100000

dataset, labels = genData(datasetSize)
# create node for input data
X = tf.placeholder("float", [None, n_visible], name='X')

# create nodes for hidden variables
W_init_max = 4 * np.sqrt(6. / (n_visible + n_hidden))
W_init = tf.random_uniform(shape=[n_visible, n_hidden],
                           minval=-W_init_max,
                           maxval=W_init_max)

W = tf.Variable(W_init, name='W')
b = tf.Variable(tf.zeros([n_hidden]), name='b')

W_prime = tf.transpose(W)  # tied weights between encoder and decoder
b_prime = tf.Variable(tf.zeros([n_visible]), name='b_prime')


        
#Z = tf.nn.relu(tf.matmul(X, W) + b)  # hidden state
#Xp = tf.nn.relu(tf.matmul(Z, W_prime) + b_prime)  # reconstructed input
Z = tf.matmul(X, W) + b # hidden state
Xp = tf.matmul(Z, W_prime) + b_prime  # reconstructed input
# create cost function
cost = tf.reduce_sum(tf.pow(X - Xp, 2))  # minimize squared error
train_op = tf.train.AdamOptimizer(0.02).minimize(cost)  # construct an optimizer


def gaussian_MLP_encoder(x, n_hidden, n_output, keep_prob):
    with tf.variable_scope("gaussian_MLP_encoder"):
        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [0][1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(x, w0) + b0
        h0 = tf.nn.elu(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd hidden layer
        w1 = tf.get_variable('w1', [0][1], n_hidden], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.tanh(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer
        wo = tf.get_variable('wo', [0][1], n_output * 2], initializer=w_init)
        bo = tf.get_variable('bo', [n_output * 2], initializer=b_init)
        gaussian_params = tf.matmul(h1, wo) + bo

        # The mean parameter is unconstrained
        mean = gaussian_params[:, :n_output]
        # The standard deviation must be positive. Parametrize with a softplus and
        # add a small epsilon for numerical stability
        stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, n_output:])

    return mean, stddev

# Bernoulli MLP as decoder
def bernoulli_MLP_decoder(z, n_hidden, n_output, keep_prob, reuse=False):

    with tf.variable_scope("bernoulli_MLP_decoder", reuse=reuse):
        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [z.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(z, w0) + b0
        h0 = tf.nn.tanh(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd hidden layer
        w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.elu(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer-mean
        wo = tf.get_variable('wo', [h1.get_shape()[1], n_output], initializer=w_init)get_variable('bo', [n_output], initializer=b_init)
        y = tf.sigmoid(tf.matmul(h1, wo) + bo)

    return y



# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(80):
        for start, end in zip(range(0, datasetSize, 128), range(128, datasetSize, 128)):
            input_ = dataset[start:end]
            weight, _ = sess.run((W, train_op), feed_dict={X: input_})
        print(i, sess.run(cost, feed_dict={X: input_}))
        print("--Weights--\n", weight)


    testData, testLabels = genData(100)
    fig = plt.figure()
    ax1 = fig.add_subplot(1,3,1)

    ax1.set_title("Input Data")
    ax1.scatter(dataset[:,0], dataset[:,1], c=labels, s=100)
    ax1.set_aspect("equal")

    ax2 = fig.add_subplot(1,3,2)
    ax2.set_title("Latency Space")
    mappedPts = sess.run(Z, feed_dict={X: testData})
    #ax2.scatter(mappedPts[:,0], mappedPts[:,1], c=testLabels, s=100)
    ax2.scatter(mappedPts, [0]*mappedPts.size, c=testLabels, s=100)

    ax3 = fig.add_subplot(1,3,3)
    ax3.set_title("Reconstruction")
    reconPts = sess.run(Xp, feed_dict={X: testData})
    ax3.scatter(reconPts[:,0], reconPts[:,1], c=testLabels, s=100)
    ax3.set_aspect("equal")
    plt.show()