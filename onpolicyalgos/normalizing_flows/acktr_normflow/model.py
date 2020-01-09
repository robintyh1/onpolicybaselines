import tensorflow as tf
import tensorflow.contrib as tc


class Model(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]


class Layer(Model):
    def __init__(self, output_dim, name='layer', layer_norm=True, num_units=3):
        super(Layer, self).__init__(name=name)
        self.output_dim = output_dim
        self.layer_norm = layer_norm
        self.num_units = num_units

    def __call__(self, input_, reuse=False):
        num_units = self.num_units
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = input_
            x = tf.layers.dense(x, num_units)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            
            x = tf.layers.dense(x, num_units)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            
            x = tf.layers.dense(x, self.output_dim, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            #x = tf.nn.relu(x)
        return x    

