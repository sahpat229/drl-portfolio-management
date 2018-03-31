"""
Use DDPG to train a stock trader based on a window of history price
"""

from __future__ import print_function, division

from model.ddpg.actor import ActorNetwork
from model.ddpg.critic import CriticNetwork
from model.ddpg.ddpg import DDPG
from model.ddpg.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise

from environment.portfolio import PortfolioEnv
from utils.data import read_stock_history, read_stock_history_csvs, normalize

import argparse
import numpy as np
import tflearn
import tensorflow as tf
import pandas as pd
import pprint
import utils.datacontainer

DEBUG = True


def get_model_path(window_length, predictor_type, use_batch_norm, learning_steps=0, gamma=0.5,
                   auxiliary_commission=0, actor_auxiliary_prediction=0, critic_auxiliary_prediction=0):
    if use_batch_norm:
        batch_norm_str = 'batch_norm'
    else:
        batch_norm_str = 'no_batch_norm'

    learning_steps_str = 'learning_steps_'+str(learning_steps)
    gamma_str = 'gamma_'+str(gamma)
    auxiliary_str = 'ac_{}_aap_{}_cap_{}'.format(str(float(auxiliary_commission)),
                                                 str(float(actor_auxiliary_prediction)),
                                                 str(float(critic_auxiliary_prediction)))

    #return 'weights/stock/{}/window_{}/{}'.format(predictor_type, window_length, batch_norm_str)

    return 'weights/{}/window_{}/{}/{}/{}/{}/'.format(predictor_type, window_length, batch_norm_str,
                                                      learning_steps_str, gamma_str, auxiliary_str)


def get_result_path(window_length, predictor_type, use_batch_norm, learning_steps=0, gamma=0.5,
                    auxiliary_commission=0, actor_auxiliary_prediction=0, critic_auxiliary_prediction=0):
    if use_batch_norm:
        batch_norm_str = 'batch_norm'
    else:
        batch_norm_str = 'no_batch_norm'

    learning_steps_str = 'learning_steps_'+str(learning_steps)
    gamma_str = 'gamma_'+str(gamma)
    auxiliary_str = 'ac_{}_aap_{}_cap_{}'.format(str(float(auxiliary_commission)),
                                                 str(float(actor_auxiliary_prediction)),
                                                 str(float(critic_auxiliary_prediction)))

    #return 'results/stock/{}/window_{}/{}'.format(predictor_type, window_length, batch_norm_str)

    return 'results/{}/window_{}/{}/{}/{}/{}/'.format(predictor_type, window_length, batch_norm_str,
                                                      learning_steps_str, gamma_str, auxiliary_str)

def get_infer_path(window_length, predictor_type, use_batch_norm, learning_steps=0, gamma=0.5,
                    auxiliary_commission=0, actor_auxiliary_prediction=0, critic_auxiliary_prediction=0):
    if use_batch_norm:
        batch_norm_str = 'batch_norm'
    else:
        batch_norm_str = 'no_batch_norm'

    learning_steps_str = 'learning_steps_'+str(learning_steps)
    gamma_str = 'gamma_'+str(gamma)
    auxiliary_str = 'ac_{}_aap_{}_cap_{}'.format(str(float(auxiliary_commission)),
                                                 str(float(actor_auxiliary_prediction)),
                                                 str(float(critic_auxiliary_prediction)))
    #return 'results/stock/{}/window_{}/{}'.format(predictor_type, window_length, batch_norm_str)

    return 'infer/{}/window_{}/{}/{}/{}/{}/'.format(predictor_type, window_length, batch_norm_str,
                                                    learning_steps_str, gamma_str, auxiliary_str)


def get_variable_scope(window_length, predictor_type, use_batch_norm, learning_steps=0, gamma=0.5,
                       auxiliary_commission=0, actor_auxiliary_prediction=0, critic_auxiliary_prediction=0):
    if use_batch_norm:
        batch_norm_str = 'batch_norm'
    else:
        batch_norm_str = 'no_batch_norm'

    learning_steps_str = 'learning_steps_'+str(learning_steps)
    gamma_str = 'gamma_'+str(gamma)
    auxiliary_str = 'ac_{}_aap_{}_cap_{}'.format(str(float(auxiliary_commission)),
                                                 str(float(actor_auxiliary_prediction)),
                                                 str(float(critic_auxiliary_prediction)))
    #return '{}_window_{}_{}'.format(predictor_type, window_length, batch_norm_str)

    return '{}_window_{}_{}_{}_{}_{}'.format(predictor_type, window_length, batch_norm_str,
                                             learning_steps_str, gamma_str, auxiliary_str)


def stock_predictor_actor(inputs, predictor_type, use_batch_norm, use_previous, previous_input,
                          actor_auxiliary_prediction, target):
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

        with tf.variable_scope("actor_auxiliary_prediction"+str(target)):
            auxiliary_prediction = None
            if actor_auxiliary_prediction > 0:
                auxiliary_prediction = tflearn.conv_2d(net, 1, (1, 1), padding='valid')
                auxiliary_prediction = tflearn.flatten(auxil)

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
        # input shape [batch_size, num_assets, window_length, num_features]
        num_stocks = inputs.get_shape()[1]
        hidden_dim = 32
        net = tf.transpose(inputs, [0, 2, 3, 1])
        resultlist = []
        reuse = False
        for i in range(num_stocks):
            if i > 0:
                reuse = True
            print("LAYER:", i)
            result = tflearn.layers.lstm(net[:, :, :, i],
                                         hidden_dim,
                                         dropout=0.5,
                                         scope="lstm_actor"+str(target),
                                         reuse=reuse)
            resultlist.append(result)
        net = tf.stack(resultlist)
        net = tf.transpose(net, [1, 0, 2])
        print("STACKED Shape:", net.shape)
        net = tf.reshape(net, [-1, int(num_stocks), 1, hidden_dim])

        with tf.variable_scope("actor_auxiliary_prediction"+str(target)):
            auxiliary_prediction = None
            if actor_auxiliary_prediction > 0:
                auxiliary_prediction = tflearn.conv_2d(net, 1, (1, 1), padding='valid')
                auxiliary_prediction = tflearn.flatten(auxil)

        if use_previous:
            net = tflearn.layers.merge_ops.merge([previous_input, net], 'concat', axis=-1)
            if DEBUG:
                print('After concat:', net.shape)
            net = tflearn.conv_2d(net, 1, (1, 1), padding='valid')
        net = tflearn.flatten(net)
        if DEBUG:
            print('Output:', net.shape)

    else:
        raise NotImplementedError

    return net, auxiliary_prediction

def stock_predictor_critic(inputs, predictor_type, use_batch_norm, use_previous, previous_input,
                           critic_auxiliary_prediction, target):
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

        with tf.variable_scope("critic_auxiliary_prediction"+str(target)):
            auxiliary_prediction = None
            if critic_auxiliary_prediction > 0:
                auxiliary_prediction = tflearn.conv_2d(net, 1, (1, 1), padding='valid')
                auxiliary_prediction = tflearn.flatten(auxil)

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
        # input shape [batch_size, num_assets, window_length, num_features]
        num_stocks = inputs.get_shape()[1]
        hidden_dim = 32
        net = tf.transpose(inputs, [0, 2, 3, 1])
        resultlist = []
        reuse = False
        for i in range(num_stocks):
            if i > 0:
                reuse = True
            print("Layer:", i)
            result = tflearn.layers.lstm(net[:, :, :, i],
                                         hidden_dim,
                                         dropout=0.5,
                                         scope="lstm_critic"+str(target),
                                         reuse=reuse)
            resultlist.append(result)
        net = tf.stack(resultlist)
        net = tf.transpose(net, [1, 0, 2])
        net = tf.reshape(net, [-1, int(num_stocks), 1, hidden_dim])

        with tf.variable_scope("critic_auxiliary_prediction"+str(target)):
            auxiliary_prediction = None
            if critic_auxiliary_prediction > 0:
                auxiliary_prediction = tflearn.conv_2d(net, 1, (1, 1), padding='valid')
                auxiliary_prediction = tflearn.flatten(auxil)

        if use_previous:
            net = tflearn.layers.merge_ops.merge([previous_input, net], 'concat', axis=-1)
            if DEBUG:
                print('After concat:', net.shape)
            net = tflearn.conv_2d(net, 1, (1, 1), padding='valid')
        net = tflearn.flatten(net)
        if DEBUG:
            print('Output:', net.shape)

    else:
        raise NotImplementedError

    return net, auxiliary_prediction

class StockActor(ActorNetwork):
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size,
                 predictor_type, use_batch_norm, use_previous=False, auxiliary_commission=0,
                 actor_auxiliary_prediction=0):
        self.predictor_type = predictor_type
        self.use_batch_norm = use_batch_norm
        self.use_previous = use_previous
        self.auxiliary_commission = auxiliary_commission
        self.actor_auxiliary_prediction = actor_auxiliary_prediction
        ActorNetwork.__init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size)

    def create_actor_network(self, target):
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

        net, auxil = stock_predictor_actor(inputs, self.predictor_type, self.use_batch_norm, 
                                           self.use_previous, portfolio_reshaped, self.actor_auxiliary_prediction,
                                           target)
        out = tf.nn.softmax(net)
        scaled_out = tf.multiply(out, self.action_bound)

        loss = None
        future_y_inputs = None
        if self.actor_auxiliary_prediction > 0:
            future_y_inputs = tflearn.input_data(shape=[None] + self.a_dim, name='portfolio_input')
            loss = self.actor_auxiliary_prediction* \
                tf.reduce_mean(tf.reduce_sum(tf.square(auxil - future_y_inputs), axis=-1))

        return inputs, out, scaled_out, portfolio_inputs, loss, future_y_inputs

    def train(self, inputs, a_gradient, portfolio_inputs=None, future_y_inputs=None):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        if not self.use_previous:
            self.sess.run([self.optimize], feed_dict={
                self.inputs: inputs,
                self.action_gradient: a_gradient
            })
        else:
            if self.actor_auxiliary_prediction > 0:
                self.sess.run([self.optimize, self.optimize_comm, self.optimize_prediction], feed_dict={
                    self.inputs: inputs,
                    self.portfolio_inputs: portfolio_inputs,
                    self.action_gradient: a_gradient,
                    self.future_y_inputs: future_y_inputs
                })
            else:                
                self.sess.run([self.optimize, self.optimize_comm], feed_dict={
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
                 predictor_type, use_batch_norm, use_previous=False, critic_auxiliary_prediction=0):
        self.predictor_type = predictor_type
        self.use_batch_norm = use_batch_norm
        self.use_previous = use_previous
        self.critic_auxiliary_prediction = critic_auxiliary_prediction
        CriticNetwork.__init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars)

    def create_critic_network(self, target):
        inputs = tflearn.input_data(shape=[None] + self.s_dim + [1])
        action = tflearn.input_data(shape=[None] + self.a_dim)

        portfolio_inputs = None
        portfolio_reshaped = None
        if self.use_previous:
            portfolio_inputs = tflearn.input_data(shape=[None] + self.a_dim, name='portfolio_input')
            portfolio_reshaped = tflearn.reshape(portfolio_inputs, new_shape=[-1]+self.a_dim+[1, 1])

        net, auxil = stock_predictor_critic(inputs, self.predictor_type, self.use_batch_norm, 
                                            self.use_previous, portfolio_reshaped, self.critic_auxiliary_prediction,
                                            target)

        loss = 0
        future_y_inputs = None
        if self.critic_auxiliary_prediction > 0:
            future_y_inputs = tflearn.input_data(shape=[None] + self.a_dim, name='portfolio_input')
            loss = self.critic_auxiliary_prediction* \
                tf.reduce_mean(tf.reduce_sum(tf.square(auxil - future_y_inputs), axis=-1))

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
        return inputs, action, out, portfolio_inputs, loss, future_y_inputs

    def train(self, inputs, action, predicted_q_value, portfolio_inputs=None, future_y_inputs=None):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        if not self.use_previous:
            return self.sess.run([self.out, self.optimize], feed_dict={
                self.inputs: inputs,
                self.action: action,
                self.predicted_q_value: predicted_q_value
            })
        else:
            if self.critic_auxiliary_prediction > 0:
                return self.sess.run([self.out, self.optimize], feed_dict={
                    self.inputs: inputs,
                    self.portfolio_inputs: portfolio_inputs,
                    self.action: action,
                    self.predicted_q_value: predicted_q_value,
                    self.future_y_inputs: future_y_inputs
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
    # divisor = observation[:, -1, 3]
    # divisor = divisor[:, None, None]
    # observation = observation[:, :, 1:4] / divisor
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

    parser.add_argument('--debug', '-d', help='print debug statement', default=False, type=bool)
    parser.add_argument('--predictor_type', '-p', help='cnn or lstm predictor', required=True)
    parser.add_argument('--window_length', '-w', help='observation window length', required=True, type=int)
    parser.add_argument('--batch_norm', '-b', help='whether to use batch normalization', required=True, type=bool)
    parser.add_argument('--learning_steps', '-l', help='number of learning steps for DDPG', required=True, type=int)
    parser.add_argument('--auxil_commission', '-ac', help='whether to use auxiliary commission', default=0, type=float)
    parser.add_argument('--actor_auxil_prediction', '-aap', help='whether to use actor auxiliary prediction', default=0, type=float)
    parser.add_argument('--critic_auxil_prediction', '-ap', help='whether to use critic_auxiliary prediction', default=0, type=float)
    parser.add_argument('--actor_tau', '-at', help='actor tau constant', default=1e-3, type=float)
    parser.add_argument('--critic_tau', '-ct', help='critic tau constant', default=1e-3, type=float)
    parser.add_argument('--actor_learning_rate', '-al', help='actor learning rate', default=1e-4, type=float)
    parser.add_argument('--critic_learning_rate', '-cl', help='critic learning rate', default=1e-3, type=float)
    parser.add_argument('--batch_size', '-bs', help='batch size', default=64, type=int)
    parser.add_argument('--action_bound', '-ab', help='action bound', default=1, type=int)
    parser.add_argument('--load_weights', '-lw', help='load previous weights', default=False, type=bool)
    parser.add_argument('--gamma', '-g', help='gamma value', default=0.5, type=float)
    parser.add_argument('--training_episodes', '-te', help='number of episodes to train on', default=600, type=int)
    parser.add_argument('--max_rollout_steps', '-mre', help='number of steps to rollout in an episode', default=1000, type=int)
    parser.add_argument('--buffer_size', '-bus', help='replay buffer size', default=100000, type=int)
    parser.add_argument('--seed', '-s', help='seed value', default=1337, type=int)

    args = vars(parser.parse_args())

    pprint.pprint(args)

    DEBUG=args['debug']
    predictor_type = args['predictor_type']
    window_length = args['window_length']
    use_batch_norm = args['batch_norm']
    learning_steps = args['learning_steps']
    auxil_commission = args['auxil_commission']
    actor_auxil_prediction = args['actor_auxil_prediction']
    critic_auxil_prediction = args['critic_auxil_prediction']
    actor_tau = args['actor_tau']
    critic_tau = args['critic_tau']
    actor_learning_rate = args['actor_learning_rate']
    critic_learning_rate = args['critic_learning_rate']
    batch_size = args['batch_size']
    action_bound = args['action_bound']
    load_weights = args['load_weights']
    gamma = args['gamma']
    training_episodes = args['training_episodes']
    max_rollout_steps = args['max_rollout_steps']
    buffer_size = args['buffer_size']
    seed = args['seed']

    assert args['predictor_type'] in ['cnn', 'lstm'], 'Predictor must be either cnn or lstm'

##################################### NASDAQ ##########################################

    history, abbreviation = read_stock_history(filepath='utils/datasets/stocks_history_target.h5')
    history = history[:, :, :4]
    #history[:, 1:, 0] = history[:, 0:-1, 3] # correct opens
    target_stocks = abbreviation
    num_training_time = 1095

    # get target history
    target_history = np.empty(shape=(len(target_stocks), num_training_time, history.shape[2]))
    for i, stock in enumerate(target_stocks):
        target_history[i] = history[abbreviation.index(stock), :num_training_time, :]
    print("target:", target_history.shape)

    testing_stocks = abbreviation
    test_history = np.empty(shape=(len(testing_stocks), history.shape[1] - num_training_time,
                                   history.shape[2]))
    for i, stock in enumerate(testing_stocks):
        test_history[i] = history[abbreviation.index(stock), num_training_time:, :]
    print("test:", test_history.shape)

################################## JIANG DATA ##########################################

    # history = np.load('history.pkl')
    # history = np.transpose(history, [1, 2, 0])
    # closes = history[:, :, 0]
    # opens = closes[:, :-1]
    # closes = closes[:, 1:]
    # history = np.stack((opens, history[:, 1:, 1], history[:, 1:, 2], closes), axis=-1)
    # print("sHAPE:", history.shape)

    # num_training_time = int(history.shape[1] * 8 / 9)
    # stocks = ['' for _ in range(history.shape[0])]
    # target_stocks = stocks
    # testing_stocks = stocks

    # # get target history
    # target_history = np.empty(shape=(len(target_stocks), num_training_time, history.shape[2]))
    # for i, stock in enumerate(target_stocks):
    #     target_history[i] = history[i, :num_training_time, :]
    # print("target:", target_history.shape)

    # test_history = np.empty(shape=(len(testing_stocks), history.shape[1] - num_training_time,
    #                                history.shape[2]))
    # for i, stock in enumerate(testing_stocks):
    #     test_history[i] = history[i, num_training_time:, :]
    # print("test:", test_history.shape)

################################## DOW JONES ###########################################
    # history, abbreviation = read_stock_history_csvs(csv_directory='./datasets/')
    # history = history[:, :, :4]
    # #history[:, 1:, 2] = history[:, 0:-1, 3] # correct opens
    # target_stocks = abbreviation
    # num_training_time = int(history.shape[1] * 3 / 4)

    # # get target history
    # target_history = np.empty(shape=(len(target_stocks), num_training_time, history.shape[2]))
    # for i, stock in enumerate(target_stocks):
    #     target_history[i] = history[abbreviation.index(stock), :num_training_time, :]

    # testing_stocks = abbreviation
    # test_history = np.empty(shape=(len(testing_stocks), history.shape[1] - num_training_time,
    #                                history.shape[2]))
    # for i, stock in enumerate(testing_stocks):
    #     test_history[i] = history[abbreviation.index(stock), num_training_time:, :]

######################################## BITCOIN #######################################

    # pd_data = pd.read_hdf('./datasets/poloniex_30m.hf', key='train')
    # asset_names = list(pd_data.columns.levels[0])
    # closes = [pd_data[asset_name, 'close'].values[::48] for asset_name in asset_names]
    # opens = [pd_data[asset_name, 'open'].values[::48] for asset_name in asset_names]
    # lows = [pd_data[asset_name, 'low'].values[::48] for asset_name in asset_names]
    # highs = [pd_data[asset_name, 'high'].values[::48] for asset_name in asset_names]
    # target_history = np.stack([opens, highs, lows, closes], axis=-1)

    # pd_data = pd.read_hdf('./datasets/poloniex_30m.hf', key='test')
    # asset_names = list(pd_data.columns.levels[0])
    # closes = [pd_data[asset_name, 'close'].values[::48] for asset_name in asset_names]
    # opens = [pd_data[asset_name, 'open'].values[::48] for asset_name in asset_names]
    # lows = [pd_data[asset_name, 'low'].values[::48] for asset_name in asset_names]
    # highs = [pd_data[asset_name, 'high'].values[::48] for asset_name in asset_names]
    # test_history = np.stack([opens, highs, lows, closes], axis=-1)

    # target_stocks = asset_names
    # testing_stocks = asset_names

######################################## TEST CONTAINER ########################################

    # setup environment

    # dc = utils.datacontainer.TestContainer(shape='ar', num_assets=4, num_samples=20000, alpha=0.9, kappa=3)
    # #dc = utils.datacontainer.BitcoinTestContainer(csv_file_name='./datasets/output.csv')
    # target_history = dc.train_close
    # num_assets = target_history.shape[0]
    # opens = target_history[:, :-1]
    # target_history = target_history[:, 1:]
    # filler = np.zeros((num_assets, target_history.shape[1]))
    # target_history = np.stack((opens, filler, filler, target_history), axis=-1)
    # target_stocks = ['' for _ in range(4)]

    # test_history = dc.test_close
    # num_assets = test_history.shape[0]
    # opens = test_history[:, :-1]
    # test_history = test_history[:, 1:]
    # filler = np.zeros((num_assets, test_history.shape[1]))
    # test_history = np.stack((opens, filler, filler, test_history), axis=-1)
    # testing_stocks = target_stocks

###############################################################################################
    train_env = PortfolioEnv(target_history, 
                             target_stocks, 
                             steps=min(max_rollout_steps, target_history.shape[1]-window_length-learning_steps), 
                             window_length=window_length)
    infer_train_env = PortfolioEnv(target_history, 
                                   target_stocks, 
                                   steps=target_history.shape[1]-window_length-learning_steps,
                                   window_length=window_length)
    infer_test_env = PortfolioEnv(test_history, 
                                  testing_stocks, 
                                  steps=test_history.shape[1]-window_length-learning_steps, 
                                  window_length=window_length)
    infer_train_env.reset()
    infer_test_env.reset()
    nb_classes = len(target_stocks) + 1

    action_dim = [nb_classes]
    state_dim = [nb_classes, window_length]

    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))
    model_save_path = get_model_path(window_length, predictor_type, use_batch_norm, 
                                     learning_steps, gamma, auxil_commission, actor_auxil_prediction,
                                     critic_auxil_prediction)
    summary_path = get_result_path(window_length, predictor_type, use_batch_norm,
                                   learning_steps, gamma, auxil_commission, actor_auxil_prediction,
                                   critic_auxil_prediction)
    infer_path = get_infer_path(window_length, predictor_type, use_batch_norm,
                                learning_steps, gamma, auxil_commission, actor_auxil_prediction,
                                critic_auxil_prediction)
    variable_scope = get_variable_scope(window_length, predictor_type, use_batch_norm,
                                        learning_steps, gamma, auxil_commission, actor_auxil_prediction,
                                        critic_auxil_prediction)

    with tf.variable_scope(variable_scope):
        sess = tf.Session()
        actor = StockActor(sess=sess, state_dim=state_dim, action_dim=action_dim, action_bound=action_bound, 
                           learning_rate=1e-4, tau=actor_tau, batch_size=batch_size,
                           predictor_type=predictor_type, use_batch_norm=use_batch_norm, use_previous=True,
                           auxiliary_commission=auxil_commission, actor_auxiliary_prediction=actor_auxil_prediction)
        critic = StockCritic(sess=sess, state_dim=state_dim, action_dim=action_dim, tau=critic_tau,
                             learning_rate=1e-3, num_actor_vars=actor.get_num_trainable_vars(),
                             predictor_type=predictor_type, use_batch_norm=use_batch_norm, use_previous=True,
                             critic_auxiliary_prediction=critic_auxil_prediction)
        ddpg_model = DDPG(train_env, sess, actor, critic, actor_noise, obs_normalizer=obs_normalizer,
                          gamma=gamma, training_episodes=training_episodes, max_rollout_steps=max_rollout_steps,
                          buffer_size=buffer_size, seed=seed, batch_size=batch_size, model_save_path=model_save_path,
                          summary_path=summary_path, infer_path=infer_path, infer_train_env=infer_train_env,
                          infer_test_env=infer_test_env, learning_steps=learning_steps)
        ddpg_model.initialize(load_weights=load_weights, verbose=False)
        ddpg_model.train()
