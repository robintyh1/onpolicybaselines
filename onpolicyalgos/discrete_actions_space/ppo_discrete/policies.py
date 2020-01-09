import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, ortho_init, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype

class MlpDiscretePolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False): #pylint: disable=W0613
        ob_shape = (nbatch,) + ob_space.shape
        #actdim = ac_space.shape[0]
        self.pdtype = pdtype = make_pdtype(ac_space)
        X = tf.placeholder(tf.float32, ob_shape, name='Ob') #obs
        with tf.variable_scope("model", reuse=reuse):
            h1 = fc(X, 'pi_fc1', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            h2 = fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            pdparam = fc(h2, 'pi', pdtype.param_shape()[0], act=lambda x:x, init_scale=0.01)
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


# =====
class MlpInterpolatedDiscretePolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False): #pylint: disable=W0613
        ob_shape = (nbatch,) + ob_space.shape
        #actdim = ac_space.shape[0]
        self.pdtype = pdtype = make_pdtype(ac_space)
        X = tf.placeholder(tf.float32, ob_shape, name='Ob') #obs
        bins = (ac_space.high - ac_space.low)[0] + 1
        # atomic actions
        atomic_actions = np.linspace(-1, 1, bins) # default [-1,1]
        with tf.variable_scope("model", reuse=reuse):
            h1 = fc(X, 'pi_fc1', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            h2 = fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            # concat actions to the hidden vectors
            logits = []
            #for idx,action in enumerate(atomic_actions):
            #    action_tf = tf.ones(shape=[h2.get_shape().as_list()[0], 1], dtype=tf.float32) * action
            #    h2_prime = tf.concat([h2, action_tf], axis=1)
            #    logit = fc(h2_prime, 'logit_{}'.format(idx), ac_space.low.size, act=lambda x:x, init_scale=0.01)
            #    logits.append(logit) 
            
            # need a correct order
            # so that categoricalpd can unzip the parameter properly (in the right order)
            for idx in range(ac_space.low.size): # act_dim
                logit_dimension = [] # logit for each dimension
                for idxa,a in enumerate(atomic_actions):
                    action_tf = tf.ones(shape=[h2.get_shape().as_list()[0], 1], dtype=tf.float32) * a
                    h2_prime = tf.concat([h2, action_tf], axis=1)
                    logit = fc(h2_prime, 'logit_{}_{}'.format(idx, idxa), 1, act=lambda x:x, init_scale=0.01)
                    logit_dimension.append(logit)
                logits_dimension_tf = tf.concat(logit_dimension, axis=1) # [None, bins]
                logits.append(logits_dimension_tf)
            
            # we want to reuse parameters for a single action dimension
            # do not create a new set of network for each bin, use the same parameter but the input is now different
            #for idx in range(ac_space.low.size): # act_dim
            #    logit_dimension = [] # logit for each dimension
            #    with tf.variable_scope('logitparam_{}'.format(idx)):
            #        nin = h2.get_shape()[1].value + 1
            #        init_scale=0.01
            #        w = tf.get_variable("w", [nin, 1], initializer=ortho_init(init_scale))
            #        b = tf.get_variable("b", [1], initializer=tf.constant_initializer(0.0))
            #        for idxa,a in enumerate(atomic_actions):
            #            action_tf = tf.ones(shape=[h2.get_shape().as_list()[0], 1], dtype=tf.float32) * a
            #            h2_prime = tf.concat([h2, action_tf], axis=1)
            #            logit = tf.matmul(h2_prime, w) + b
            #            logit_dimension.append(logit)
            #        logits_dimension_tf = tf.concat(logit_dimension, axis=1) # [None, bins]
            #    logits.append(logits_dimension_tf)
            pdparam = tf.concat(logits, axis=1) # [None, bins*actdim]
            #print(pdparam)
            #hi
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
