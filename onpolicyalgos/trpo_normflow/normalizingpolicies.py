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

class TRPOImplicitPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, num_units=3, num_layers=4):
        assert isinstance(ob_space, gym.spaces.Box)

        nbatch_train = 1024
        nbatch_vf_train = 64
        nbatch_fvp_train = 205 # sub-sampled size
        self.ob_train = ob_train = U.get_placeholder(name="ob_train", dtype=tf.float32, shape=[nbatch_train] + list(ob_space.shape))
        self.action_train = action_train = U.get_placeholder(name='ac_train', dtype=tf.float32, shape=[nbatch_train] + list(ac_space.shape))
        ob_act = U.get_placeholder(name="ob_act", dtype=tf.float32, shape=[1] + list(ob_space.shape))
        action_act = U.get_placeholder(name='ac_act', dtype=tf.float32, shape=[1] + list(ac_space.shape))
        self.ob_vf_train = ob_vf_train = U.get_placeholder(name="ob_vf_train", dtype=tf.float32, shape=[nbatch_vf_train] + list(ob_space.shape))
        self.ob_fvp_train = ob_fvp_train = U.get_placeholder(name="ob_fvp_train", dtype=tf.float32, shape=[nbatch_fvp_train] + list(ob_space.shape))
        self.ac_fvp_train = action_fvp_train = U.get_placeholder(name="ac_fvp_act", dtype=tf.float32, shape=[nbatch_fvp_train] + list(ac_space.shape))
        
        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        obz_train = tf.clip_by_value((ob_train - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        obz_act = tf.clip_by_value((ob_act - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        obz_vf_train = tf.clip_by_value((ob_vf_train - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        obz_fvp_train = tf.clip_by_value((ob_fvp_train - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

        # value function
        last_out = obz_vf_train
        with tf.variable_scope('value', reuse=False):
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(U.dense(last_out, hid_size, "vffc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
            self.vpred_train = U.dense(last_out, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:,0]
        last_out = obz_act
        with tf.variable_scope('value', reuse=True):
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(U.dense(last_out, hid_size, "vffc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
            self.vpred_act = U.dense(last_out, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:,0]
        
        # policy
        policy_train = NormalizingFlowStateModel(obz_train, action_train, name='policy', reuse=False, num_units=num_units, num_layers=num_layers)
        policy_act = NormalizingFlowStateModel(obz_act, action_act, name='policy', reuse=True, num_units=num_units, num_layers=num_layers)
        policy_fvp_train = NormalizingFlowStateModel(obz_fvp_train, action_fvp_train, name='policy', reuse=True, num_units=num_units, num_layers=num_layers)
        self.pi_act = policy_act.y_sample  #act for forward sampling
        self.pi_train = policy_fvp_train.y_sample  #for fvp
        self.entropy_train = policy_train.entropy
        self.log_prob_act = policy_act.log_prob
        self.action_act = action_act
        self.log_prob_train = policy_train.log_prob  #logprob
        self.log_prob_fvp_train = policy_fvp_train.log_prob        
        
        self.state_in = []
        self.state_out = []

        #stochastic = tf.placeholder(dtype=tf.bool, shape=())
        #ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        #self._act = U.function([stochastic, ob], [ac, self.vpred])
        self._act = U.function([ob_act], [self.pi_act, self.vpred_act])
        self.ob_act = ob_act

    def act(self, stochastic, ob):
        #ac1, vpred1 =  self._act(stochastic, ob[None])
        ac1, vpred1 = self._act(ob[None])
        return ac1[0], vpred1[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []