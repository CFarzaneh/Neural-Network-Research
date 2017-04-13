import tensorflow as tf
import numpy as np
from IPython import embed
import sys

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

mnist_width = 28
n_visible = mnist_width * mnist_width
n_hidden = 100
corruption_level = 0.3

# create node for input data
X = tf.placeholder("float", [None, n_visible], name='X')

# create node for corruption mask
mask = tf.placeholder("float", [None, n_visible], name='mask')

# create nodes for hidden variables
W_init_max = 4 * np.sqrt(6. / (n_visible + n_hidden))
W_init = tf.random_uniform(shape=[n_visible, n_hidden],
                           minval=-W_init_max,
                           maxval=W_init_max)

W = tf.Variable(W_init, name='W')
b = tf.Variable(tf.zeros([n_hidden]), name='b')

W_prime = tf.transpose(W)  # tied weights between encoder and decoder
b_prime = tf.Variable(tf.zeros([n_visible]), name='b_prime')


def model(X, mask, W, b, W_prime, b_prime):
    tilde_X = mask * X  # corrupted X

    Z = tf.nn.sigmoid(tf.matmul(tilde_X, W) + b)  # hidden state
    Xp = tf.nn.sigmoid(tf.matmul(Z, W_prime) + b_prime)  # reconstructed input
    return Xp

# build model graph
Xp = model(X, mask, W, b, W_prime, b_prime)

# create cost function
cost = tf.reduce_sum(tf.pow(X - Xp, 2))  # minimize squared error
train_op = tf.train.GradientDescentOptimizer(0.02).minimize(cost)  # construct an optimizer

# load MNIST data
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

reconX = None
tildX = None
# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    for i in range(10):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
            input_ = trX[start:end]
            embed()
            sys.exit()
            mask_np = np.random.binomial(1, 1 - corruption_level, input_.shape)
            sess.run(train_op, feed_dict={X: input_, mask: mask_np})

        mask_np = np.random.binomial(1, 1 - corruption_level, teX.shape)
        print(i, sess.run(cost, feed_dict={X: teX, mask: mask_np}))
    m = sess.run(mask, feed_dict={X: trX[0:1], mask: mask_np})
    #embed()
    tildX = m[0] *trX[0]
    reconX = sess.run(Xp, feed_dict={X: trX[0:1], mask: mask_np})


import matplotlib.pyplot as plt
plt.figure(0); plt.imshow(trX[0].reshape(28,28))
plt.figure(1); plt.imshow(tildX.reshape(28,28))
plt.figure(2); plt.imshow(reconX[0].reshape(28,28))
plt.show()
