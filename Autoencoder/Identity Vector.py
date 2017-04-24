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
#embed()
#sys.exit()
#plt.scatter(dataset[:,0], dataset[:,1], c=labels, s=100)
#plt.show()
#sys.exit()
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
train_op = tf.train.AdagradOptimizer(0.02).minimize(cost)  # construct an optimizer

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



