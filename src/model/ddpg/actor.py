"""
Actor Network definition, The CNN architecture follows the one in this paper
https://arxiv.org/abs/1706.10059
Author: Patrick Emami, Modified by Chi Zhang
"""

import tensorflow as tf


# ===========================
#   Actor DNNs
# ===========================

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
        """

        Args:
            sess: a tensorflow session
            state_dim: a list specifies shape
            action_dim: a list specified action shape
            action_bound: whether to normalize action in the end
            learning_rate: learning rate
            tau: target network update parameter
            batch_size: use for normalization
        """
        self.sess = sess
        assert isinstance(state_dim, list), 'state_dim must be a list.'
        self.s_dim = state_dim
        assert isinstance(action_dim, list), 'action_dim must be a list.'
        self.a_dim = action_dim
        print("Self.a_dim:", self.a_dim)
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        # Actor Network
        self.inputs, self.out, self.scaled_out, self.portfolio_inputs, \
            self.loss, self.future_y_inputs = self.create_actor_network(False)

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out, \
            self.target_portfolio_inputs, self.target_loss, self.target_future_y_inputs = self.create_actor_network(True)

        self.target_network_params = tf.trainable_variables()[
                                     len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        action_grad_dim = [self.a_dim[0]+1]
        self.action_gradient = tf.placeholder(tf.float32, [None] + action_grad_dim)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        actor_grad_params = [v for v in self.network_params if "auxilFalse" not in v.name]
        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, actor_grad_params, -self.action_gradient)

        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = optimizer.apply_gradients(zip(self.actor_gradients, actor_grad_params))

        print("AUXIL PREDICTION:", self.auxiliary_prediction)
        if self.auxiliary_prediction > 0:
            mse_diff = tf.reduce_mean(tf.reduce_sum(tf.square(self.scaled_out - self.portfolio_inputs), axis=-1))
            self.optimize_comm = tf.train.AdamOptimizer(self.learning_rate).minimize(loss=self.auxiliary_prediction*self.loss,
                                                                                     var_list=self.network_params)
            print("HERE")
            #self.optimize_comm = tf.train.AdamOptimizer(self.learning_rate).minimize(loss=self.auxiliary_prediction*self.loss,
            #                                                                         var_list=self.network_params)

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        raise NotImplementedError('Create actor should return (inputs, out, scaled_out)')

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars
