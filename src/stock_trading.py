"""
Use DDPG to train a stock trader based on a window of history price
"""

from __future__ import print_function, division

from model.ddpg.actor import ActorNetwork
from model.ddpg.critic import CriticNetwork
from model.ddpg.ddpg import DDPG
from model.ddpg.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise

from environment.portfolio import PortfolioEnv
from utils.data import read_stock_history, normalize

import numpy as np
import tflearn
import tensorflow as tf
import argparse
import pprint
import utils.datacontainer

DEBUG = True


def get_model_path(window_length, predictor_type, use_batch_norm):
    if use_batch_norm:
        batch_norm_str = 'batch_norm'
    else:
        batch_norm_str = 'no_batch_norm'
    return 'weights/stock/{}/window_{}/{}/checkpoint.ckpt'.format(predictor_type, window_length, batch_norm_str)


def get_result_path(window_length, predictor_type, use_batch_norm):
    if use_batch_norm:
        batch_norm_str = 'batch_norm'
    else:
        batch_norm_str = 'no_batch_norm'
    return 'results/stock/{}/window_{}/{}/'.format(predictor_type, window_length, batch_norm_str)


def get_variable_scope(window_length, predictor_type, use_batch_norm):
    if use_batch_norm:
        batch_norm_str = 'batch_norm'
    else:
        batch_norm_str = 'no_batch_norm'
    return '{}_window_{}_{}'.format(predictor_type, window_length, batch_norm_str)


def stock_predictor(inputs, predictor_type, use_batch_norm, use_previous, previous_input):
    window_length = inputs.get_shape()[2]
    assert predictor_type in ['cnn', 'lstm'], 'type must be either cnn or lstm'
    if predictor_type == 'cnn':
        net = tflearn.conv_2d(inputs, 32, (1, 3), padding='valid')
        if use_batch_norm:
            net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.conv_2d(net, 32, (1, window_length - 2), padding='valid')
        if use_batch_norm:
            net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        if DEBUG:
            print('After conv2d:', net.shape)
        if use_previous:
            net = tflearn.layers.merge_ops.merge([previous_input, net], 'concat', axis=-1)
            if DEBUG:
                print('After concat:', net.shape)
            net = tflearn.conv_2d(net, 1, (1, 1), padding='valid')
            if DEBUG:
                print('After portfolio conv2d:', net.shape)
        net = tflearn.flatten(net)
        if DEBUG:
            print('Output:', net.shape)
    elif predictor_type == 'lstm':
        num_stocks = inputs.get_shape()[1]
        hidden_dim = 32
        net = tflearn.reshape(inputs, new_shape=[-1, window_length, 1])
        if DEBUG:
            print('Reshaped input:', net.shape)
        net = tflearn.lstm(net, hidden_dim)
        if DEBUG:
            print('After LSTM:', net.shape)
        net = tflearn.reshape(net, new_shape=[-1, num_stocks, hidden_dim])
        if DEBUG:
            print('After reshape:', net.shape)
        net = tflearn.flatten(net)
        if DEBUG:
            print('Output:', net.shape)
    else:
        raise NotImplementedError

    return net


class StockActor(ActorNetwork):
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size,
                 predictor_type, use_batch_norm, use_previous=False):
        self.predictor_type = predictor_type
        self.use_batch_norm = use_batch_norm
        self.use_previous = use_previous
        ActorNetwork.__init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size)

    def create_actor_network(self):
        """
        self.s_dim: a list specifies shape
        """
        nb_classes, window_length = self.s_dim
        assert nb_classes == self.a_dim[0]
        assert window_length > 2, 'This architecture only support window length larger than 2.'
        inputs = tflearn.input_data(shape=[None] + self.s_dim + [1], name='input')

        portfolio_inputs = None
        portfolio_reshaped = None
        if self.use_previous:
            portfolio_inputs = tflearn.input_data(shape=[None] + self.a_dim, name='portfolio_input')
            portfolio_reshaped = tflearn.reshape(portfolio_inputs, new_shape=[-1]+self.a_dim+[1, 1])

        net = stock_predictor(inputs, self.predictor_type, self.use_batch_norm, self.use_previous, portfolio_reshaped)

        net = tflearn.fully_connected(net, 64)
        if self.use_batch_norm:
            net = tflearn.layers.normalization.batch_normalization(net)
        # net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 64)
        if self.use_batch_norm:
            net = tflearn.layers.normalization.batch_normalization(net)
        # net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, self.a_dim[0], activation='softmax', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out, portfolio_inputs

    def train(self, inputs, a_gradient, portfolio_inputs=None):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        if not self.use_previous:
            self.sess.run(self.optimize, feed_dict={
                self.inputs: inputs,
                self.action_gradient: a_gradient
            })
        else:
            self.sess.run(self.optimize, feed_dict={
                self.inputs: inputs,
                self.portfolio_inputs: portfolio_inputs,
                self.action_gradient: a_gradient
            })

    def predict(self, inputs, portfolio_inputs=None):
        #print("BEFOREINPUT:", inputs.shape)
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        #print("INPUTS:", inputs.shape)
        if not self.use_previous:
            return self.sess.run(self.scaled_out, feed_dict={
                self.inputs: inputs
            })
        else:
            return self.sess.run(self.scaled_out, feed_dict={
                self.inputs:inputs,
                self.portfolio_inputs: portfolio_inputs
            })

    def predict_target(self, inputs, portfolio_inputs=None):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        if not self.use_previous:
            return self.sess.run(self.target_scaled_out, feed_dict={
                self.target_inputs: inputs
            })
        else:
            return self.sess.run(self.target_scaled_out, feed_dict={
                self.target_inputs: inputs,
                self.target_portfolio_inputs: portfolio_inputs
            })

class StockCritic(CriticNetwork):
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars,
                 predictor_type, use_batch_norm, use_previous=False):
        self.predictor_type = predictor_type
        self.use_batch_norm = use_batch_norm
        self.use_previous = use_previous
        CriticNetwork.__init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None] + self.s_dim + [1])
        action = tflearn.input_data(shape=[None] + self.a_dim)

        portfolio_inputs = None
        portfolio_reshaped = None
        if self.use_previous:
            portfolio_inputs = tflearn.input_data(shape=[None] + self.a_dim, name='portfolio_input')
            portfolio_reshaped = tflearn.reshape(portfolio_inputs, new_shape=[-1]+self.a_dim+[1, 1])

        net = stock_predictor(inputs, self.predictor_type, self.use_batch_norm, self.use_previous, portfolio_reshaped)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 64)
        t2 = tflearn.fully_connected(action, 64)

        net = tf.add(t1, t2)
        if self.use_batch_norm:
            net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out, portfolio_inputs

    def train(self, inputs, action, predicted_q_value, portfolio_inputs=None):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        if not self.use_previous:
            return self.sess.run([self.out, self.optimize], feed_dict={
                self.inputs: inputs,
                self.action: action,
                self.predicted_q_value: predicted_q_value
            })
        else:
            return self.sess.run([self.out, self.optimize], feed_dict={
                self.inputs: inputs,
                self.portfolio_inputs: portfolio_inputs,
                self.action: action,
                self.predicted_q_value: predicted_q_value
            })

    def predict(self, inputs, action, portfolio_inputs=None):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        if not self.use_previous:
            return self.sess.run(self.out, feed_dict={
                self.inputs: inputs,
                self.action: action
            })
        else:
            return self.sess.run(self.out, feed_dict={
                self.inputs: inputs,
                self.portfolio_inputs: portfolio_inputs,
                self.action: action
            })

    def predict_target(self, inputs, action, portfolio_inputs=None):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        if not self.use_previous:
            return self.sess.run(self.target_out, feed_dict={
                self.target_inputs: inputs,
                self.target_action: action
            })
        else:
            return self.sess.run(self.target_out, feed_dict={
                self.target_inputs: inputs,
                self.target_portfolio_inputs: portfolio_inputs,
                self.target_action: action
            })

    def action_gradients(self, inputs, actions, portfolio_inputs=None):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        if not self.use_previous:
            return self.sess.run(self.action_grads, feed_dict={
                self.inputs: inputs,
                self.action: actions
            })
        else:
            return self.sess.run(self.action_grads, feed_dict={
                self.inputs: inputs,
                self.portfolio_inputs: portfolio_inputs,
                self.action: actions
            }) 


def obs_normalizer(observation):
    """ Preprocess observation obtained by environment

    Args:
        observation: (nb_classes, window_length, num_features) or with info

    Returns: normalized

    """
    if isinstance(observation, tuple):
        observation = observation[0]
    # directly use close/open ratio as feature
    observation = observation[:, :, 3:4] / observation[:, :, 0:1]
    observation = normalize(observation)
    return observation


def test_model(env, model):
    observation, info = env.reset()
    done = False
    while not done:
        action = model.predict_single(observation)
        observation, _, done, _ = env.step(action)
    env.render()


def test_model_multiple(env, models):
    observation, info = env.reset()
    done = False
    while not done:
        actions = []
        for model in models:
            actions.append(model.predict_single(observation))
        actions = np.array(actions)
        observation, _, done, info = env.step(actions)
    env.render()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Provide arguments for training different DDPG models')

    parser.add_argument('--debug', '-d', help='print debug statement', default=False)
    parser.add_argument('--predictor_type', '-p', help='cnn or lstm predictor', required=True)
    parser.add_argument('--window_length', '-w', help='observation window length', required=True)
    parser.add_argument('--batch_norm', '-b', help='whether to use batch normalization', required=True)

    args = vars(parser.parse_args())

    pprint.pprint(args)

    if args['debug'] == 'True':
        DEBUG = True
    else:
        DEBUG = False

    history, abbreviation = read_stock_history(filepath='utils/datasets/stocks_history_target.h5')
    history = history[:, :, :4]
    target_stocks = abbreviation
    num_training_time = 1095
    window_length = int(args['window_length'])
    nb_classes = len(target_stocks) + 1

    # get target history
    target_history = np.empty(shape=(len(target_stocks), num_training_time, history.shape[2]))
    for i, stock in enumerate(target_stocks):
        target_history[i] = history[abbreviation.index(stock), :num_training_time, :]
    print(target_history.shape)

    # setup environment

    #dc = utils.datacontainer.TestContainer(shape='ar', num_assets=4, num_samples=2000, alpha=0.9, kappa=3)
    # dc = utils.datacontainer.BitcoinTestContainer(csv_file_name='./datasets/output.csv')
    # target_history = dc.train_close
    # num_assets = target_history.shape[0]
    # opens = np.concatenate((np.ones((num_assets, 1)), target_history[:, :-1]), axis=1)
    # filler = np.zeros((num_assets, target_history.shape[1]))
    # target_history = np.stack((opens, filler, filler, target_history), axis=-1)
    # print(target_history.shape)
    # target_stocks = ['BTC']
    # nb_classes = len(target_stocks) + 1

    env = PortfolioEnv(target_history, target_stocks, steps=1000, window_length=window_length)

    action_dim = [nb_classes]
    state_dim = [nb_classes, window_length]
    batch_size = 64
    action_bound = 1.
    tau = 1e-3
    assert args['predictor_type'] in ['cnn', 'lstm'], 'Predictor must be either cnn or lstm'
    predictor_type = args['predictor_type']
    if args['batch_norm'] == 'True':
        use_batch_norm = True
    elif args['batch_norm'] == 'False':
        use_batch_norm = False
    else:
        raise ValueError('Unknown batch norm argument')
    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))
    model_save_path = get_model_path(window_length, predictor_type, use_batch_norm)
    summary_path = get_result_path(window_length, predictor_type, use_batch_norm)

    variable_scope = get_variable_scope(window_length, predictor_type, use_batch_norm)

    with tf.variable_scope(variable_scope):
        sess = tf.Session()
        actor = StockActor(sess, state_dim, action_dim, action_bound, 1e-4, tau, batch_size,
                           predictor_type, use_batch_norm, use_previous=True)
        critic = StockCritic(sess=sess, state_dim=state_dim, action_dim=action_dim, tau=1e-3,
                             learning_rate=1e-3, num_actor_vars=actor.get_num_trainable_vars(),
                             predictor_type=predictor_type, use_batch_norm=use_batch_norm, use_previous=True)
        ddpg_model = DDPG(env, sess, actor, critic, actor_noise, obs_normalizer=obs_normalizer,
                          config_file='config/stock.json', model_save_path=model_save_path,
                          summary_path=summary_path)
        ddpg_model.initialize(load_weights=False)
        ddpg_model.train()
