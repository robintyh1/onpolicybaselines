import numpy as np
import tensorflow as tf
from utils import fc
import gym
import baselines.common.tf_util as U
from baselines.common.mpi_running_mean_std import RunningMeanStd

tfd = tf.contrib.distributions


class GMMStateModel(object):
    """
    GMM policy base model
    K: number of cluters
    """
    def __init__(self, state, action, name, reuse, K=2, num_hid_layers=2, hid_size=32):

        input_dim = action.get_shape().as_list()[1] # mujoco: not atari
        self.input_dim = input_dim
        self.name = name
        self.state = state
        self.action = action
        self.reuse = reuse
        self.num_hid_layers = num_hid_layers
        self.hid_size = hid_size
        self.K = K
        with tf.variable_scope(name) as scope:
            print('build forward')
            self.build_forward(state, reuse=reuse)
            print('build inverse')
            self.build_inverse(action, state, reuse=reuse)
            print('build entropy')
            self.build_entropy()

    def build_forward(self, state, reuse):
    	# build noise samples
        batch_size = [state.get_shape().as_list()[0], self.input_dim]
        noise_dist = tfd.Normal(loc=0., scale=1.)
        noise_samples = noise_dist.sample(batch_size) # size of [batchsize, action dim]
        # build forward
        last_out = state
        self.meandict = meandict = []
        self.logstddict = logstddict = []
        with tf.variable_scope('forward', reuse=reuse):
            for i in range(self.num_hid_layers):
                last_out = tf.nn.tanh(U.dense(last_out, self.hid_size, "polfc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
            for k in range(self.K):
                mean = U.dense(last_out, self.input_dim, "polfinal_{}".format(k), U.normc_initializer(0.01))            
                logstd = tf.get_variable(name="logstd_{}".format(k), shape=[1, self.input_dim], initializer=tf.zeros_initializer())
                meandict.append(mean)
                logstddict.append(logstd)
        meandicttf = tf.concat(meandict, axis=1) # size of [batchsize, action dim * K]
        logstddicttf = tf.concat(logstddict, axis=1)
        # generate masks
        logits = [0.0] * self.K
        num_samples = self.state.shape.as_list()[0]
        categorical_mask = tf.multinomial([logits], num_samples)
        #print('categoricalmask', categorical_mask)
        onehot_mask = tf.squeeze(tf.one_hot(categorical_mask, self.K), 0)
        #print('onehotmask', onehot_mask)
        onehot_mask_tiled = tf.squeeze(tf.reshape(tf.tile(tf.expand_dims(onehot_mask,axis=2),[1,1,self.input_dim]),[-1,self.input_dim * self.K, 1]),axis=2)
        # select
        mean_tiled = tf.multiply(onehot_mask_tiled, meandicttf)  # size of [batchsize, action dim * K]
        logstd_tiled = tf.multiply(onehot_mask_tiled, logstddicttf)
        # sample action mean and logstd
        mean = tf.reshape(mean_tiled, [-1, self.K, self.input_dim])  # size of [batchsize, K, action dim]
        logstd = tf.reshape(logstd_tiled, [-1, self.K, self.input_dim])
        mean_final = tf.reduce_sum(mean, axis=1, keepdims=True)  # size of [batchsize, action dim]            
        logstd_final = tf.reduce_sum(logstd, axis=1, keepdims=True)
        # sample action
        action = tf.exp(logstd_final) * noise_samples + mean_final
        self.y_sample = action
        
    def build_inverse(self, action, state, reuse):
        y_ph = action
        self.y_ph = y_ph
        # compute gaussian pdf for each cluster
        logpdfdict = []
        for mean, logstd in zip(self.meandict, self.logstddict):
            std = tf.exp(logstd)
            neglogpdf = 0.5 * tf.reduce_sum(tf.square((y_ph - mean) / std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.to_float(self.input_dim) \
               + tf.reduce_sum(logstd, axis=-1) # size of [batchsize]
            logpdf = -neglogpdf
            logpdfdict.append(tf.expand_dims(logpdf, axis=1))
        logpdftf = tf.concat(logpdfdict, axis=1) # size of [batchsize, K]
        log_prob = tf.reduce_mean(logpdftf, axis=1)
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
        # compute gaussian pdf for each cluster
        y_ph = self.y_sample
        logpdfdict = []
        for mean, logstd in zip(self.meandict, self.logstddict):
            std = tf.exp(logstd)
            logpdf = 0.5 * tf.reduce_sum(tf.square((y_ph - mean) / std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.to_float(self.input_dim) \
               + tf.reduce_sum(logstd, axis=-1) # size of [batchsize]
            logpdfdict.append(logpdf)
        logpdftf = tf.concat(logpdfdict, axis=1) # size of [batchsize, K]
        log_prob = tf.reduce_mean(logpdftf, axis=1)
        self.entropy = -tf.reduce_mean(log_prob)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


# === trpo ===
class TRPOGMMPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, K):
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
        policy_train = GMMStateModel(obz_train, action_train, name='policy', reuse=False, K=K, num_hid_layers=num_hid_layers, hid_size=hid_size)
        policy_act = GMMStateModel(obz_act, action_act, name='policy', reuse=True, K=K, num_hid_layers=num_hid_layers, hid_size=hid_size)
        policy_fvp_train = GMMStateModel(obz_fvp_train, action_fvp_train, name='policy', reuse=True, K=K, num_hid_layers=num_hid_layers, hid_size=hid_size)
        self.pi_act = policy_act.y_sample  #act for forward sampling
        self.pi_train = policy_fvp_train.y_sample  #for fvp
        self.entropy_train = policy_train.entropy
        self.log_prob_train = policy_train.log_prob  #logprob
        self.log_prob_fvp_train = policy_fvp_train.log_prob        
        
        self.state_in = []
        self.state_out = []

        #stochastic = tf.placeholder(dtype=tf.bool, shape=())
        #ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        #self._act = U.function([stochastic, ob], [ac, self.vpred])
        self._act = U.function([ob_act], [self.pi_act, self.vpred_act])

    def act(self, stochastic, ob):
        #ac1, vpred1 =  self._act(stochastic, ob[None])
        if np.ndim(ob) == 1:
            ob = np.expand_dims(ob, axis=0)
        ac1, vpred1 = self._act(ob)
        return ac1[0], vpred1[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []