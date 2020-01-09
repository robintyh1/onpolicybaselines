import numpy as np
import tensorflow as tf
from baselines.acktr.utils import conv, fc, dense, conv_to_fc, sample, kl_div
import baselines.common.tf_util as U
from baselines.common.distributions import make_pdtype


class MultiCategoricalPolicy(object):
    def __init__(self, ob_dim, ac_dim, ac_space, bins):
        # Here we'll construct a bunch of expressions, which will be used in two places:
        # (1) When sampling actions
        # (2) When computing loss functions, for the policy update
        # Variables specific to (1) have the word "sampled" in them,
        # whereas variables specific to (2) have the word "old" in them
        ob_no = tf.placeholder(tf.float32, shape=[None, ob_dim*2], name="ob") # batch of observations
        oldac_na = tf.placeholder(tf.int32, shape=[None, ac_dim], name="ac") # batch of actions previous actions
        oldac_logits = tf.placeholder(tf.float32, shape=[None, ac_dim*bins], name="oldac_logit") # batch of actions previous action distributions
        adv_n = tf.placeholder(tf.float32, shape=[None], name="adv") # advantage function estimate
        self.pdtype = make_pdtype(ac_space)
        wd_dict = {}
        # forward pass
        h1 = tf.nn.tanh(dense(ob_no, 64, "h1", weight_init=U.normc_initializer(1.0), bias_init=0.0, weight_loss_dict=wd_dict))
        h2 = tf.nn.tanh(dense(h1, 64, "h2", weight_init=U.normc_initializer(1.0), bias_init=0.0, weight_loss_dict=wd_dict))
        logits_na = dense(h2, self.pdtype.param_shape()[0], "logits", weight_init=U.normc_initializer(0.1), bias_init=0.0, weight_loss_dict=wd_dict) # Mean control 
        self.wd_dict = wd_dict
        self.pd = self.pdtype.pdfromflat(logits_na) # multi-categorical distributions
        # sample action for control
        sampled_ac_na = self.pd.sample()
        # log prob for sampled actions
        logprobsampled_n = -self.pd.neglogp(sampled_ac_na)
        logprob_n = -self.pd.neglogp(oldac_na)
        # kl div
        old_pd = self.pdtype.pdfromflat(oldac_logits)
        kl = U.mean(old_pd.kl(self.pd))
        # surr loss
        surr = -U.mean(adv_n * logprob_n)
        surr_sampled = -U.mean(logprob_n)
        # expressions
        self._act = U.function([ob_no], [sampled_ac_na, logits_na, logprobsampled_n])
        self.compute_kl = U.function([ob_no, oldac_logits], kl)
        self.update_info = ((ob_no, oldac_na, adv_n), surr, surr_sampled)
        U.initialize()

        #logstd_1a = tf.expand_dims(logstd_1a, 0)
        #std_1a = tf.exp(logstd_1a)
        #std_na = tf.tile(std_1a, [tf.shape(mean_na)[0], 1])
        #ac_dist = tf.concat([tf.reshape(mean_na, [-1, ac_dim]), tf.reshape(std_na, [-1, ac_dim])], 1)
        #sampled_ac_na = tf.random_normal(tf.shape(ac_dist[:,ac_dim:])) * ac_dist[:,ac_dim:] + ac_dist[:,:ac_dim] # This is the sampled action we'll perform.
        
        #logprobsampled_n = - U.sum(tf.log(ac_dist[:,ac_dim:]), axis=1) - 0.5 * tf.log(2.0*np.pi)*ac_dim - 0.5 * U.sum(tf.square(ac_dist[:,:ac_dim] - sampled_ac_na) / (tf.square(ac_dist[:,ac_dim:])), axis=1) # Logprob of sampled action
        #logprob_n = - U.sum(tf.log(ac_dist[:,ac_dim:]), axis=1) - 0.5 * tf.log(2.0*np.pi)*ac_dim - 0.5 * U.sum(tf.square(ac_dist[:,:ac_dim] - oldac_na) / (tf.square(ac_dist[:,ac_dim:])), axis=1) # Logprob of previous actions under CURRENT policy (whereas oldlogprob_n is under OLD policy)
        #kl = U.mean(kl_div(oldac_dist, ac_dist, ac_dim))
        #kl = .5 * U.mean(tf.square(logprob_n - oldlogprob_n)) # Approximation of KL divergence between old policy used to generate actions, and new policy used to compute logprob_n
        #surr = - U.mean(adv_n * logprob_n) # Loss function that we'll differentiate to get the policy gradient
        #surr_sampled = - U.mean(logprob_n) # Sampled loss of the policy
        #self._act = U.function([ob_no], [sampled_ac_na, ac_dist, logprobsampled_n]) # Generate a new action and its logprob
        #self.compute_kl = U.function([ob_no, oldac_na, oldlogprob_n], kl) # Compute (approximate) KL divergence between old policy and new policy
        #self.compute_kl = U.function([ob_no, oldac_dist], kl)
        #self.update_info = ((ob_no, oldac_na, adv_n), surr, surr_sampled) # Input and output variables needed for computing loss
        #U.initialize() # Initialize uninitialized TF variables

    def act(self, ob):
        ac, logit, logp = self._act(ob[None])
        return ac[0], logit[0], logp[0]
