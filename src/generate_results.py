from stock_trading import get_model_path, get_variable_scope, test_model_multiple, obs_normalizer, \
    StockActor, StockCritic
from environment.portfolio import MultiActionPortfolioEnv
from model.ddpg.ddpg import DDPG
from utils.data import read_stock_history, read_stock_history_csvs, normalize

import numpy as np
import tensorflow as tf
import tflearn
import matplotlib.pyplot as plt

history, abbreviation = read_stock_history(filepath='utils/datasets/stocks_history_target.h5')
history = history[:, :, :4]
#history[:, 1:, 0] = history[:, 0:-1, 3] # correct opens
target_stocks = abbreviation
num_training_time = 1095

# get target history
target_history = np.empty(shape=(len(target_stocks), num_training_time, history.shape[2]))
for i, stock in enumerate(target_stocks):
    target_history[i] = history[abbreviation.index(stock), :num_training_time, :]

testing_stocks = abbreviation
test_history = np.empty(shape=(len(testing_stocks), history.shape[1] - num_training_time,
                               history.shape[2]))
for i, stock in enumerate(testing_stocks):
    test_history[i] = history[abbreviation.index(stock), num_training_time:, :]

nb_classes = 17
batch_size = 64

window_length_list = [3]
predictor_type_list = ['cnn', 'lstm']
batch_norm_list = [True]
learning_steps_list = [1]
gamma_list = [0.25]
auxil_tuples_list = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.1), (0.0, 0.0, 1.0),
    (0.0, 0.1, 0.0), (0.0, 0.1, 0.1), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), 
    (0.0, 0.0, 10.0), (0.0, 10.0, 0.0), (0.0, 1.0, 1.0)]
#auxil_tuples_list = [(0.0, 0.0, 0.0)]

models = []
model_names = []
for window_length in window_length_list:
    for predictor_type in predictor_type_list:
        for batch_norm in batch_norm_list:
            for learning_steps in learning_steps_list:
                for gamma in gamma_list:
                    for auxil_tuple in auxil_tuples_list:
                        # name = 'AC: {} AAP: {} CAP: {}'.format(auxil_tuple[0],
                        #                                        auxil_tuple[1],
                        #                                        auxil_tuple[2])
                        name = 'Predictor Type {}, Learning Steps {}, Window Length {}, ' \
                            'Gamma {} Auxil Tuple {}'.format(predictor_type,
                                                   learning_steps,
                                                   window_length,
                                                   gamma,
                                                   str(auxil_tuple))
                        # name = 'Window {}'.format(window_length)
                        # name = 'Gamma {}'.format(gamma)
                        model_names.append(name)
                        tf.reset_default_graph()
                        sess = tf.Session()
                        tflearn.config.init_training_mode()

                        action_dim = [nb_classes]
                        state_dim = [nb_classes, window_length]
                        variable_scope = get_variable_scope(window_length, predictor_type, batch_norm,
                                                            learning_steps, gamma, auxil_tuple[0], auxil_tuple[1],
                                                            auxil_tuple[2])
                        model_save_path = get_model_path(window_length, predictor_type, batch_norm,
                                                         learning_steps, gamma, auxil_tuple[0], auxil_tuple[1],
                                                         auxil_tuple[2])
                        with tf.variable_scope(variable_scope):
                            actor = StockActor(sess=sess, state_dim=state_dim, action_dim=action_dim, action_bound=1, 
                                               learning_rate=1e-4, tau=1e-3, batch_size=batch_size,
                                               predictor_type=predictor_type, use_batch_norm=batch_norm, use_previous=True,
                                               auxiliary_commission=auxil_tuple[0], actor_auxiliary_prediction=auxil_tuple[1])
                            critic = StockCritic(sess=sess, state_dim=state_dim, action_dim=action_dim, tau=1e-3,
                                                 learning_rate=1e-3, num_actor_vars=actor.get_num_trainable_vars(),
                                                 predictor_type=predictor_type, use_batch_norm=batch_norm, use_previous=True,
                                                 critic_auxiliary_prediction=auxil_tuple[2])
                            ddpg_model = DDPG(None, sess, actor, critic, None, obs_normalizer=obs_normalizer,
                                              gamma=gamma, training_episodes=None, max_rollout_steps=None,
                                              buffer_size=None, seed=None, batch_size=batch_size, model_save_path=model_save_path,
                                              summary_path=None, infer_path=None, infer_train_env=None,
                                              infer_test_env=None, learning_steps=None)
                            ddpg_model.initialize(load_weights=True, verbose=False)
                            models.append(ddpg_model)

print("ALL models:", len(models))

env = MultiActionPortfolioEnv(test_history, test_history, model_names, steps=test_history.shape[1]-50-2, 
                              window_length=50)
test_model_multiple(env, models)
env.make_df()
# dic_results = np.load('algos.npy')[()]
# #print("DIC RESULTS:", dic_results)
# for result in dic_results:
#     env.model_names.append(result)
#     env.df_info[result] = dic_results[result]
#env.render()
#plt.savefig('rendered.png', bbox_inches='tight')
#plt.close()

stats = env.stats()
for name in model_names:
    print(name, stats[name])