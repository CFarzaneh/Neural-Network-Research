import tensorflow as tf
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

n_hidden = 1

X = tf.placeholder("float", name='X')

# create nodes for hidden variables
W_init_max = 4 * np.sqrt(6. / (dataset))
W_init = tf.random_uniform(shape=dataset,
                           minval=-W_init_max,
                           maxval=W_init_max)

W = tf.Variable(W_init, name='W')
b = tf.Variable(tf.zeros([n_hidden]), name='b')

W_prime = tf.transpose(W)  # tied weights between encoder and decoder
b_prime = tf.Variable(tf.zeros(dataset), name='b_prime') # want to pass dataset in to get 0s?

def model(X, mask, W, b, W_prime, b_prime):

    Z = tf.nn.sigmoid(W + b)  # hidden state
    Xp = tf.nn.sigmoid(tf.matmul(Z, W_prime) + b_prime)  # reconstructed input
    return Xp

Xp = model(X, mask, W, b, W_prime, b_prime)

# create cost function
cost = tf.reduce_sum(tf.pow(X - Xp, 2))  # minimize squared error
train_op = tf.train.GradientDescentOptimizer(0.02).minimize(cost)  # construct an optimizer

with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    for i in range(10):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
            input_ = trX[start:end]
            mask_np = np.random.binomial(1, 1 - corruption_level, input_.shape)
            sess.run(train_op, feed_dict={X: input_, mask: mask_np})

        mask_np = np.random.binomial(1, 1 - corruption_level, teX.shape)
        print(i, sess.run(cost, feed_dict={X: teX, mask: mask_np}))
    m = sess.run(mask, feed_dict={X: trX[0:1], mask: mask_np})
    #embed()
    tildX = m[0] *trX[0]
    reconX = sess.run(Xp, feed_dict={X: trX[0:1], mask: mask_np})