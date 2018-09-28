import numpy as np
import tensorflow as tf
from normalizingflow_model import make_mask, NormalizingFlowLayer
from model import Layer
from utils import fc

tfd = tf.contrib.distributions

ENTROPY_NUMSAMPLES = 10

class NormalizingFlowStateModel(object):
    """
    Normalizingflow state conditional model
    accept state from outside computation graph
    Args:
        state_step: state tf variable for forward sampling (batch dimension cannot be None)
        state_train: state tf variable for backward training (batch dimension cannot be None)
        action: action tf variable for backward training
        name: name of the model
        reuse: if to reuse model
        num_units: num of hidden unit in s,t net
        num_layers: num of alternating units
    """
    def __init__(self, state, action, name, reuse, num_units=3, num_layers=4):

        input_dim = action.get_shape().as_list()[1] # mujoco: not atari
        self.input_dim = input_dim
        m1, m2 = make_mask(input_dim)
        self.mask = [m1, m2] * num_layers
        print(self.mask)
        self.name = name
        self.state = state
        self.action = action
        self.reuse = reuse
        self.num_units = num_units

        with tf.variable_scope(name) as scope:
            self.build_forward(state, reuse=reuse)
            self.build_inverse(action, state, reuse=reuse)
            self.build_entropy()

    def build_forward(self, state, reuse):
    	# build noise samples
        batch_size = [state.get_shape().as_list()[0], self.input_dim]
        noise_dist = tfd.Normal(loc=0., scale=1.)
        noise_samples = noise_dist.sample(batch_size)
        # build noise
        x_ph = noise_samples
        s_ph = state
        self.x_ph = x_ph
        self.s_ph = s_ph
        x = x_ph
        layers = []
        for i in range(len(self.mask)):
            layer = NormalizingFlowLayer(self.input_dim, self.mask[i], name='Nlayer_{}'.format(i), num_units=self.num_units)
            x = layer.forward(x, reuse=reuse)
            if i == 0: # fuse state information into the first layer
                self.statelayer = Layer(self.input_dim, name='statelayer', num_units=64)
                s_processed = self.statelayer(s_ph, reuse=reuse)
                x += s_processed
            layers.append(layer)
        self.y_sample = x
        self.layers = layers
        
    def build_inverse(self, action, state, reuse):
        y_ph = action
        self.y_ph = y_ph
        self.state = state
        y = y_ph
        log_det_sum = 0
        for i, layer in enumerate(self.layers[::-1]):
            if i == len(self.layers) - 1: # subtract state component
                s_processed = self.statelayer(state, reuse=True)  # we are sure to reuse
                y -= s_processed
            y, logdet = layer.inverse(y)
            log_det_sum += logdet
        self.x_sample = y
        # build objective
        # sample x initially drawn from a gaussian with mean 0 and std 1
        log_prior = -tf.reduce_sum(tf.square(self.x_sample), axis=1) / 2.0
        log_det = log_det_sum
        log_prob = log_prior - log_det
        self.log_prior = log_prior
        self.log_prob = log_prob

    def build_entropy(self):
        """
        build entropy to allow for gradient computation of gradient
        a bit different from conventional methods since we use new seeds (instead of 
        sampled actions, since that way we need to track the original seeds and generate
        old actions)
        """
        # sample only one action per state seems to be of high variance
        # sample multiple actions per state
        # build noise samples
        batch_size = [self.state.get_shape().as_list()[0] * ENTROPY_NUMSAMPLES, self.input_dim]
        noise_dist = tfd.Normal(loc=0., scale=1.)
        noise_samples = noise_dist.sample(batch_size)        

        # build noise
        x_ph = noise_samples
        s_ph = tf.tile(self.state, [ENTROPY_NUMSAMPLES, 1])  # replicate states
        x = x_ph
        for i,layer in enumerate(self.layers):
            x = layer.forward(x, reuse=True)  # we are sure to reuse
            if i == 0: # fuse state information into the first layer
                s_processed = self.statelayer(s_ph, reuse=True)
                x += s_processed
        y_sample = x        

        y_ph = y_sample  # action from forward sampling
        y = y_ph
        log_det_sum = 0
        for i, layer in enumerate(self.layers[::-1]):
            if i == len(self.layers) - 1: # subtract state component
                s_processed = self.statelayer(s_ph, reuse=True)  # we are sure to reuse
                y -= s_processed
            y, logdet = layer.inverse(y)
            log_det_sum += logdet
        x_sample = y
        log_prior = -tf.reduce_sum(tf.square(x_sample), axis=1) / 2.0
        log_det = log_det_sum
        log_prob = log_prior - log_det
        self.entropy = -tf.reduce_mean(log_prob)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


def create_mlp(X, outputdim, reuse=False, name=None):
    assert name is not None
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        h1 = fc(X, '{}_fc1'.format(name), nh=64, init_scale=np.sqrt(2), act=tf.tanh)
        h2 = fc(h1, '{}_fc2'.format(name), nh=64, init_scale=np.sqrt(2), act=tf.tanh)
        vf = fc(h2, name, outputdim)
    return vf


# ===========
# === trpo ===
# ===========
from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype

class ACKTRImplicitPolicy(object):

    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_dim, act_dim, num_units=3, num_layers=4, batch=None):
        assert batch is not None
        self.batch = batch
        ob_act = tf.placeholder(tf.float32, shape=[1, ob_dim*2], name="ob_act")
        ob_train = tf.placeholder(tf.float32, shape=[batch, ob_dim*2], name="ob_train")
        oldac_na = tf.placeholder(tf.float32, shape=[batch, act_dim], name="ac")
        action_act = tf.placeholder(tf.float32, shape=[1, act_dim], name="ac_act")
        oldac_dist = tf.placeholder(tf.float32, shape=[batch], name="oldac_dist") # logprob for old actions
        adv_n = tf.placeholder(tf.float32, shape=[batch], name="adv")
        wd_dict = {}

        # module for execution and training
        policy_train = NormalizingFlowStateModel(ob_train, oldac_na, name='policy', reuse=False, num_units=num_units, num_layers=num_layers)
        policy_act = NormalizingFlowStateModel(ob_act, action_act, name='policy', reuse=True, num_units=num_units, num_layers=num_layers)        

        # weight decay
        self.wd_dict = {} # TODO

        # action for execution
        self.pi_act = policy_act.y_sample
        self.log_prob_act = policy_act.log_prob
        
        # kl divergence
        ac_dist = policy_train.log_prob # logprob
        kl = U.mean(oldac_dist - ac_dist) # sample based kl

        # surr loss
        surr = - U.mean(adv_n * ac_dist)
        surr_sampled = - U.mean(ac_dist)

        # functions
        self._act = U.function([ob_act], self.pi_act)
        self._act_logprob = U.function([ob_act, action_act], self.log_prob_act)
        self.compute_kl = U.function([ob_train, oldac_na, oldac_dist], kl)
        self.update_info = ((ob_train, oldac_na, adv_n), surr, surr_sampled)
        U.initialize()

    def act(self, ob):
        ac = self._act(ob[None])
        logp = self._act_logprob(ob[None], ac)
        return ac[0], logp[0]
