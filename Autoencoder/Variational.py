import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from IPython import embed
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def genData(numPts):

    u1 = np.array((1,1))/np.sqrt(2)
    u2 = np.array((-1,1))/np.sqrt(2)
    basis = [u1, u2]

    magnitude = 3

    labels = np.random.binomial(1, 0.5, numPts)
    scales = np.random.uniform(-magnitude, magnitude, numPts)

    dataset = np.array([scale*basis[index] for (scale,index) in
        zip(scales,labels)], dtype='float32')

    return (dataset, labels)

