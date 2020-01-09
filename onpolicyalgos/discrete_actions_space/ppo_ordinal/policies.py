import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, ortho_init, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype

from ordinal_utils import action_mask, construct_mask

class MlpDiscretePolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False): #pylint: disable=W0613
        bins = ac_space.high[0] - ac_space.low[0] + 1
        print('making policy bins size {}'.format(bins))
        assert bins is not None
        ob_shape = (nbatch,) + ob_space.shape
        #actdim = ac_space.shape[0]
        self.pdtype = pdtype = make_pdtype(ac_space)
        X = tf.placeholder(tf.float32, ob_shape, name='Ob') #obs
        with tf.variable_scope("model", reuse=reuse):
            h1 = fc(X, 'pi_fc1', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            h2 = fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            #pdparam = fc(h2, 'pi', pdtype.param_shape()[0], act=lambda x:x, init_scale=0.01)
            m = fc(h2, 'pi', pdtype.param_shape()[0], act=lambda x:x, init_scale=0.01) # of size [batchsize, num-actions*bins]
            norm_softm = tf.nn.sigmoid(m) # of size [batchsize, num-actions*bins], initialized to be about uniform
            norm_softm = tf.reshape(norm_softm, [nbatch,-1,bins]) # of size [batchsize, num-actions, bins], initialized to be about uniform

            norm_softm_tiled = tf.tile(tf.expand_dims(norm_softm, axis=-1), [1,1,1,bins])

            # construct the mask
            am_numpy = construct_mask(bins)
            am_tf = tf.constant(am_numpy, dtype=tf.float32)

            # construct pdparam
            pdparam = tf.reduce_sum(tf.math.log(norm_softm_tiled + 1e-8) * am_tf + tf.math.log(1 - norm_softm_tiled + 1e-8) * (1 - am_tf), axis=-1)
            pdparam = tf.reshape(pdparam, [nbatch, -1])
            
            # value function
            h1 = fc(X, 'vf_fc1', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            h2 = fc(h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            vf = fc(h2, 'vf', 1, act=lambda x:x)[:,0]
            #logstd = tf.get_variable(name="logstd", shape=[1, actdim], 
            #    initializer=tf.zeros_initializer())

        #pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.pdparam = pdparam
        self.vf = vf
        self.step = step
        self.value = value



def fc(x, scope, nh, act=tf.nn.relu, init_scale=1.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(0.0))
        z = tf.matmul(x, w)+b
        h = act(z)
        return h
