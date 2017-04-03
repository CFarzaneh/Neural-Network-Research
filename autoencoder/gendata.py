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
indices = np.random.binomial(1, 0.5, datasetSize)
# scale is an array with 'datasetSize' floats between
# 0 and magnitude. These are used for scaling the vectors
scales = np.random.uniform(0, magnitude, datasetSize)

# construct the dataset by scaling each of the randomly selected
# basis vectors. We're only using list comprehension here. There is
# probablyi a slicker way of doing this in the future
dataset = [scale*basis[index] for (scale,index) in zip(scales,indices)]

print("dataset = ",dataset)

X = tf.placeholder("float", [None, magnitude], name='X')

