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

import argparse
import numpy as np
import tflearn
import tensorflow as tf
import pandas as pd
import pprint
import utils.datacontainer

DEBUG = True


def get_model_path(window_length, predictor_type, use_batch_norm, learning_steps=0, 
                   auxiliary_commission=False, auxiliary_prediction=False):
    if use_batch_norm:
        batch_norm_str = 'batch_norm'
    else:
        batch_norm_str = 'no_batch_norm'

    learning_steps_str = 'learning_steps_'+str(learning_steps)
    auxiliary_str = 'auxil_commission_{}_auxil_prediction_{}'.format(str(auxiliary_commission), str(auxiliary_prediction))

    #return 'weights/stock/{}/window_{}/{}'.format(predictor_type, window_length, batch_norm_str)

    return 'weights/stock/{}/window_{}/{}/{}/{}/'.format(predictor_type, window_length, batch_norm_str,
                                                         learning_steps_str, auxiliary_str)


def get_result_path(window_length, predictor_type, use_batch_norm, learning_steps=0,
                    auxiliary_commission=False, auxiliary_prediction=False):
    if use_batch_norm:
        batch_norm_str = 'batch_norm'
    else:
        batch_norm_str = 'no_batch_norm'

    learning_steps_str = 'learning_steps_'+str(learning_steps)
    auxiliary_str = 'auxil_commission_{}_auxil_prediction_{}'.format(str(auxiliary_commission), str(auxiliary_prediction))

    #return 'results/stock/{}/window_{}/{}'.format(predictor_type, window_length, batch_norm_str)

    return 'results/stock/{}/window_{}/{}/{}/{}/'.format(predictor_type, window_length, batch_norm_str,
                                                         learning_steps_str, auxiliary_str)

def get_infer_path(window_length, predictor_type, use_batch_norm, learning_steps=0,
                    auxiliary_commission=False, auxiliary_prediction=False):
    if use_batch_norm:
        batch_norm_str = 'batch_norm'
    else:
        batch_norm_str = 'no_batch_norm'

    learning_steps_str = 'learning_steps_'+str(learning_steps)
    auxiliary_str = 'auxil_commission_{}_auxil_prediction_{}'.format(str(auxiliary_commission), str(auxiliary_prediction))

    #return 'results/stock/{}/window_{}/{}'.format(predictor_type, window_length, batch_norm_str)

    return 'infer/stock/{}/window_{}/{}/{}/{}/'.format(predictor_type, window_length, batch_norm_str,
                                                         learning_steps_str, auxiliary_str)


def get_variable_scope(window_length, predictor_type, use_batch_norm, learning_steps=0,
                       auxiliary_commission=False, auxiliary_prediction=False):
    if use_batch_norm:
        batch_norm_str = 'batch_norm'
    else:
        batch_norm_str = 'no_batch_norm'

    learning_steps_str = 'learning_steps_'+str(learning_steps)
    auxiliary_str = 'auxil_commission_{}_auxil_prediction_{}'.format(str(auxiliary_commission), str(auxiliary_prediction))

    #return '{}_window_{}_{}'.format(predictor_type, window_length, batch_norm_str)

    return '{}_window_{}_{}_{}_{}'.format(predictor_type, window_length, batch_norm_str,
                                          learning_steps_str, auxiliary_str)


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
                 predictor_type, use_batch_norm, use_previous=False, auxiliary_prediction=False):
        self.predictor_type = predictor_type
        self.use_batch_norm = use_batch_norm
        self.use_previous = use_previous
        self.auxiliary_prediction = auxiliary_prediction
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
        out = tf.nn.softmax(net)
        scaled_out = tf.multiply(out, self.action_bound)

        # net = tflearn.fully_connected(net, 64)
        # if self.use_batch_norm:
        #     net = tflearn.layers.normalization.batch_normalization(net)
        # # net = tflearn.layers.normalization.batch_normalization(net)
        # net = tflearn.activations.relu(net)
        # net = tflearn.fully_connected(net, 64)
        # if self.use_batch_norm:
        #     net = tflearn.layers.normalization.batch_normalization(net)
        # # net = tflearn.layers.normalization.batch_normalization(net)
        # net = tflearn.activations.relu(net)
        # # Final layer weights are init to Uniform[-3e-3, 3e-3]
        # w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        # out = tflearn.fully_connected(net, self.a_dim[0], activation='softmax', weights_init=w_init)
        # # Scale output to -action_bound to action_bound
        # scaled_out = tf.multiply(out, self.action_bound)

        loss = None
        if self.use_previous and self.auxiliary_prediction:
            loss = tf.reduce_mean(tf.reduce_sum(tf.square(scaled_out - portfolio_inputs), axis=-1))

        return inputs, out, scaled_out, portfolio_inputs, loss

    def train(self, inputs, a_gradient, portfolio_inputs=None):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        if not self.use_previous:
            self.sess.run([self.optimize], feed_dict={
                self.inputs: inputs,
                self.action_gradient: a_gradient
            })
        else:
            self.sess.run([self.optimize], feed_dict={
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

    parser.add_argument('--debug', '-d', help='print debug statement', default=False, type=bool)
    parser.add_argument('--predictor_type', '-p', help='cnn or lstm predictor', required=True)
    parser.add_argument('--window_length', '-w', help='observation window length', required=True, type=int)
    parser.add_argument('--batch_norm', '-b', help='whether to use batch normalization', required=True, type=bool)
    parser.add_argument('--learning_steps', '-l', help='number of learning steps for DDPG', required=True, type=int)
    parser.add_argument('--auxil_commission', '-ac', help='whether to use auxiliary commission', default=False, type=bool)
    parser.add_argument('--auxil_prediction', '-ap', help='whether to use auxiliary prediction', default=False, type=bool)
    parser.add_argument('--actor_tau', '-at', help='actor tau constant', default=1e-3, type=float)
    parser.add_argument('--critic_tau', '-ct', help='critic tau constant', default=1e-3, type=float)
    parser.add_argument('--actor_learning_rate', '-al', help='actor learning rate', default=1e-4, type=float)
    parser.add_argument('--critic_learning_rate', '-cl', help='critic learning rate', default=1e-3, type=float)
    parser.add_argument('--batch_size', '-bs', help='batch size', default=64, type=int)
    parser.add_argument('--action_bound', '-ab', help='action bound', default=1, type=int)
    parser.add_argument('--load_weights', '-lw', help='load previous weights', default=False, type=bool)

    args = vars(parser.parse_args())

    pprint.pprint(args)

    DEBUG=args['debug']
    predictor_type = args['predictor_type']
    window_length = args['window_length']
    use_batch_norm = args['batch_norm']
    learning_steps = args['learning_steps']
    auxil_commission = args['auxil_commission']
    auxil_prediction = args['auxil_prediction']
    actor_tau = args['actor_tau']
    critic_tau = args['critic_tau']
    actor_learning_rate = args['actor_learning_rate']
    critic_learning_rate = args['critic_learning_rate']
    batch_size = args['batch_size']
    action_bound = args['action_bound']
    load_weights = args['load_weights']

    assert args['predictor_type'] in ['cnn', 'lstm'], 'Predictor must be either cnn or lstm'

    history, abbreviation = read_stock_history(filepath='utils/datasets/stocks_history_target.h5')
    history = history[:, :, :4]
    target_stocks = abbreviation
    num_training_time = 1095
    nb_classes = len(target_stocks) + 1

    # get target history
    target_history = np.empty(shape=(len(target_stocks), num_training_time, history.shape[2]))
    for i, stock in enumerate(target_stocks):
        target_history[i] = history[abbreviation.index(stock), :num_training_time, :]
    print(target_history.shape)

    testing_stocks = abbreviation
    test_history = np.empty(shape=(len(testing_stocks), history.shape[1] - num_training_time,
                                   history.shape[2]))
    for i, stock in enumerate(testing_stocks):
        test_history[i] = history[abbreviation.index(stock), num_training_time:, :]

    env = PortfolioEnv(target_history, target_stocks, steps=1000, window_length=window_length)
    test_env = PortfolioEnv(test_history, testing_stocks, steps=test_history.shape[1]-10, window_length=window_length)


    # pd_data = pd.read_hdf('./datasets/poloniex_30m.hf', key='train')
    # asset_names = list(pd_data.columns.levels[0])
    # closes = [pd_data[asset_name, 'close'].values for asset_name in asset_names]
    # opens = [pd_data[asset_name, 'open'].values for asset_name in asset_names]
    # lows = [pd_data[asset_name, 'low'].values for asset_name in asset_names]
    # highs = [pd_data[asset_name, 'high'].values for asset_name in asset_names]
    # target_history = np.stack([opens, highs, lows, closes], axis=-1)

    # pd_data = pd.read_hdf('./datasets/poloniex_30m.hf', key='test')
    # asset_names = list(pd_data.columns.levels[0])
    # closes = [pd_data[asset_name, 'close'].values for asset_name in asset_names]
    # opens = [pd_data[asset_name, 'open'].values for asset_name in asset_names]
    # lows = [pd_data[asset_name, 'low'].values for asset_name in asset_names]
    # highs = [pd_data[asset_name, 'high'].values for asset_name in asset_names]
    # test_history = np.stack([opens, highs, lows, closes], axis=-1)    

    # nb_classes = len(asset_names) + 1
    # env = PortfolioEnv(target_history, asset_names, steps=3000, window_length=window_length)
    # test_env = PortfolioEnv(test_history, asset_names, steps=3000, window_length=window_length)

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

    action_dim = [nb_classes]
    state_dim = [nb_classes, window_length]

    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))
    model_save_path = get_model_path(window_length, predictor_type, use_batch_norm, 
                                     learning_steps, auxil_commission, auxil_prediction)
    summary_path = get_result_path(window_length, predictor_type, use_batch_norm,
                                   learning_steps, auxil_commission, auxil_prediction)
    infer_path = get_infer_path(window_length, predictor_type, use_batch_norm,
                                learning_steps, auxil_commission, auxil_prediction)
    variable_scope = get_variable_scope(window_length, predictor_type, use_batch_norm,
                                        learning_steps, auxil_commission, auxil_prediction)

    with tf.variable_scope(variable_scope):
        sess = tf.Session()
        actor = StockActor(sess=sess, state_dim=state_dim, action_dim=action_dim, action_bound=action_bound, 
                           learning_rate=1e-4, tau=actor_tau, batch_size=batch_size,
                           predictor_type=predictor_type, use_batch_norm=use_batch_norm, use_previous=True)
        critic = StockCritic(sess=sess, state_dim=state_dim, action_dim=action_dim, tau=critic_tau,
                             learning_rate=1e-3, num_actor_vars=actor.get_num_trainable_vars(),
                             predictor_type=predictor_type, use_batch_norm=use_batch_norm, use_previous=True)
        ddpg_model = DDPG(env, sess, actor, critic, actor_noise, obs_normalizer=obs_normalizer,
                          config_file='config/stock.json', model_save_path=model_save_path,
                          summary_path=summary_path, infer_path=infer_path, test_env=test_env, learning_steps=learning_steps)
        ddpg_model.initialize(load_weights=load_weights, verbose=False)
        ddpg_model.train()
