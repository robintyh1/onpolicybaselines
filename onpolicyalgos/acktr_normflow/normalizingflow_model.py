from model import Layer
import numpy as np
import tensorflow as tf


def make_mask(input_dim):
    """
	input: dimension of variable
	output: two mask numpy array (with alternating masks)

	example: dim = 5
	masks will be [1,1,1,0,0] and [0,0,1,1,1]
	"""
    mask1 = np.zeros(input_dim)
    mask2 = np.zeros(input_dim)
    if input_dim % 2 == 0:
        mask1[:input_dim//2] = 1.0
        mask2[input_dim//2:] = 1.0
    else:
        mask1[:(1+input_dim)//2] = 1.0
        mask2[(1+input_dim)//2:] = 1.0
    return np.float32(mask1), np.float32(mask2)


class NormalizingFlowLayer(object):

    def __init__(self, input_dim, mask, name=None, num_units=3):
        # random mask
        assert mask.size == input_dim
        self.mask = mask
        self.d = int(np.sum(mask))
        self.input_dim = input_dim
        self.name = name
        with tf.variable_scope(name):
            self.layer_s = Layer(input_dim, name='s', num_units=num_units)
            self.layer_t = Layer(input_dim, name='t', num_units=num_units)

    def forward(self, x, reuse=False):
        with tf.variable_scope(self.name):
            layer_s, layer_t = self.layer_s, self.layer_t
            # mask the input
            x_mask = tf.multiply(x, self.mask)
            # compute forward pass in the net
            h = (tf.multiply(x, tf.exp(layer_s(x_mask, reuse=reuse))) + 
                layer_t(x_mask, reuse=reuse))
            # compute final result
            y = x_mask + tf.multiply(1.0 - self.mask, h)
        return y

    def inverse(self, y, reuse=True):
        with tf.variable_scope(self.name):
            layer_s, layer_t = self.layer_s, self.layer_t
            # mask the input
            y_mask = tf.multiply(y, self.mask)
            # compute forward pass in the net
            h = (tf.multiply(tf.exp(-layer_s(y_mask, reuse=reuse)),
                 y - layer_t(y_mask, reuse=reuse)))
            # compute final result
            x = y_mask + tf.multiply(1.0 - self.mask, h)
            # compute the jacobian determinant
            # in real nvp paper the det has exp, logdet removes the exp
            logdet = tf.reduce_sum(tf.multiply(1.0 - self.mask, layer_s(tf.multiply(x, self.mask), reuse=True)), axis=1)
        return x, logdet