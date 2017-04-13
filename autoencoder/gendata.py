import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np

# These are the basis vectors. They are unit vectors
# so we can easily control the magnitude
u1 = np.array((1,1))/np.sqrt(2)
u2 = np.array((-1,1))/np.sqrt(2)
basis = [u1, u2]

datasetSize = 10
magnitude = 3

# indices is an array with 'datasetSize' 0's and 1's
# chosen from a binomial distribution with equal probability
indices = np.random.binomial(1, 0.5, datasetSize) 						#either 0 or 1
# scale is an array with 'datasetSize' floats between
# 0 and magnitude. These are used for scaling the vectors
scales = np.random.uniform(-magnitude, magnitude, datasetSize)			#Between -3 to 3 vector

# construct the dataset by scaling each of the randomly selected
# basis vectors. We're only using list comprehension here. There is
# probablyi a slicker way of doing this in the future
dataset = [scale*basis[index] for (scale,index) in zip(scales,indices)]

print("dataset = ",dataset)

#n_visible = mnist_width * mnist_width
n_hidden = 1

# create node for input data
X = tf.placeholder("float", [None, 2], name='X')

# create nodes for hidden variables
W_init_max = 4 * np.sqrt(6. / (n_hidden))
W_init = tf.random_uniform(shape=[2, n_hidden],
                           minval=-W_init_max,
                           maxval=W_init_max)

W = tf.Variable(W_init, name='W')
b = tf.Variable(tf.zeros([n_hidden]), name='b')

W_prime = tf.transpose(W)  # tied weights between encoder and decoder
b_prime = tf.Variable(tf.zeros([0, 2]), name='b_prime')


def model(X, W, b, W_prime, b_prime):
    Z = tf.nn.sigmoid(tf.matmul(X, W) + b)  # hidden state
    Xp = tf.nn.sigmoid(tf.matmul(Z, W_prime) + b_prime)  # reconstructed input
    return Xp

# build model graph
Xp = model(X, W, b, W_prime, b_prime)

# create cost function
cost = tf.reduce_sum(tf.pow(X - Xp, 2))  # minimize squared error
train_op = tf.train.GradientDescentOptimizer(0.02).minimize(cost)  # construct an optimizer

# load MNIST data
#trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(10):
        for start, end in zip(range(0, len(dataset), 128), range(128, len(dataset), 128)):
            input_ = dataset[start:end]
            sess.run(train_op, feed_dict={X: input_})


        print(i, sess.run(cost, feed_dict={X: dataset}))