"""
The deep deterministic policy gradient model. Contains main training loop and deployment
"""
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')

import os
import traceback
import json
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from collections import deque
from copy import copy
from .replay_buffer import ReplayBuffer, ReplayBufferMultiple, ReplayBufferRollout
from ..base_model import BaseModel


def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax_Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars


class DDPG(BaseModel):
    def __init__(self, env, sess, actor, critic, actor_noise, obs_normalizer=None, action_processor=None,
                 gamma=0.5, training_episodes=600, max_rollout_steps=1000, buffer_size=100000, seed=1337, batch_size=64,
                 model_save_path='weights/ddpg/ddpg.ckpt', summary_path='results/ddpg/', infer_path='infer/',
                 infer_train_env=None, infer_test_env=None, learning_steps=1):
        np.random.seed(seed)
        if env:
            env.seed(seed)
        self.model_save_path = model_save_path
        #self.model_save_path = 'weights/stock/cnn/window_3/batch_norm/checkpoint.ckpt'
        self.summary_path = summary_path
        self.infer_path = infer_path
        self.sess = sess
        # if env is None, then DDPG just predicts
        self.env = env
        self.actor = actor
        self.critic = critic
        self.actor_noise = actor_noise
        self.obs_normalizer = obs_normalizer
        self.action_processor = action_processor
        self.gamma = gamma
        self.training_episodes = training_episodes
        self.max_rollout_steps = max_rollout_steps
        self.buffer_size = buffer_size
        self.seed = seed
        self.batch_size = batch_size
        self.infer_train_env = infer_train_env
        self.infer_test_env = infer_test_env
        self.learning_steps = learning_steps
        self.start_episode = 0
        self.summary_ops, self.summary_vars = build_summaries()

    def clear_path(self, folder):
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                #elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)

    def initialize(self, load_weights=True, verbose=True):
        """ Load training history from path. To be add feature to just load weights, not training states

        """

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path, exist_ok=True)
        if not os.path.exists(self.summary_path):
            os.makedirs(self.summary_path, exist_ok=True)
        if not os.path.exists(os.path.join(self.infer_path, 'test/')):
            os.makedirs(os.path.join(self.infer_path, 'test/'), exist_ok=True)
        if not os.path.exists(os.path.join(self.infer_path, 'train/')):
            os.makedirs(os.path.join(self.infer_path, 'train/'), exist_ok=True)

        if load_weights:
            try:
                variables = tf.global_variables()
                param_dict = {}
                saver = tf.train.Saver()
                latest_checkpoint = tf.train.latest_checkpoint(self.model_save_path)
                self.start_episode = int(latest_checkpoint.split('-')[1]) + 1
                saver.restore(self.sess, latest_checkpoint)
                for var in variables:
                    var_name = var.name[:-2]
                    if verbose:
                        print('Loading {} from checkpoint. Name: {}'.format(var.name, var_name))
                    param_dict[var_name] = var
            except:
                traceback.print_exc()
                print('Build model from scratch')
                self.sess.run(tf.global_variables_initializer())
        else:
            print('Build model from scratch')
            self.clear_path(self.model_save_path)
            self.clear_path(self.summary_path)
            self.clear_path(os.path.join(self.infer_path, 'test'))
            self.clear_path(os.path.join(self.infer_path, 'train'))
            self.sess.run(tf.global_variables_initializer())

    def train(self, save_every_episode=1, verbose=True, debug=False):
        """ Must already call intialize

        Args:
            save_every_episode:
            print_every_step:
            verbose:
            debug:

        Returns:

        """
        writer = tf.summary.FileWriter(self.summary_path, self.sess.graph)

        self.actor.update_target_network()
        self.critic.update_target_network()

        np.random.seed(self.seed)
        num_episode = self.training_episodes
        batch_size = self.batch_size
        gamma = self.gamma
        self.buffer = ReplayBufferRollout(self.buffer_size)

        # main training loop
        for i in range(self.start_episode, num_episode):
            if verbose and debug:
                print("Episode: " + str(i) + " Replay Buffer " + str(self.buffer.count()))

            episode_rollout = deque()

            observation_1 = self.env.reset()
            observation_1, weights_1 = observation_1[0]['obs'], observation_1[0]['weights']

            if self.obs_normalizer:
                observation_1 = self.obs_normalizer(observation_1)

            episode_rollout.append([observation_1, weights_1])

            for rollout_step in range(self.learning_steps - 1):
                obs, ws = episode_rollout[-1]
                action = self.actor.predict(inputs=np.expand_dims(obs, axis=0),
                                            portfolio_inputs=np.expand_dims(ws, axis=0)).squeeze(
                                            axis=0) + self.actor_noise()
                action = np.clip(action, 0, 1)
                if action.sum() == 0:
                    action = np.ones(obs.shape[0])/obs.shape[0]
                action /= action.sum()
                new_obs, reward, done, info = self.env.step(action)
                new_obs, new_ws = new_obs['obs'], new_obs['weights']

                if self.obs_normalizer:
                    new_obs = self.obs_normalizer(new_obs)
                episode_rollout.append(action)
                episode_rollout.append(reward)
                episode_rollout.append(done)
                episode_rollout.append(info['next_y1'])
                episode_rollout.append([new_obs, new_ws])

            ep_reward = 0
            ep_ave_max_q = 0
            ep_ave_min_q = 0
            # keeps sampling until done
            for j in range(self.max_rollout_steps):
                #print(j)
                action = self.actor.predict(inputs=np.expand_dims(episode_rollout[-1][0], axis=0),
                                            portfolio_inputs=np.expand_dims(episode_rollout[-1][1], axis=0)).squeeze(
                    axis=0) + self.actor_noise()

                if self.action_processor:
                    action = self.action_processor(action)
                else:
                    action = action

                action = np.clip(action, 0, 1)
                if action.sum() == 0:
                    action = np.ones(episode_rollout[-1][0].shape[0])/episode_rollout[-1][0].shape[0]
                action /= action.sum()

                obs, reward, done, info = self.env.step(action)
                obs, ws = obs['obs'], obs['weights']

                if self.obs_normalizer:
                    obs = self.obs_normalizer(obs)

                episode_rollout.append(action)
                episode_rollout.append(reward)
                episode_rollout.append(done)
                episode_rollout.append(info['next_y1'])
                episode_rollout.append([obs, ws])

                # add to buffer
                self.buffer.add(copy(episode_rollout))

                if self.buffer.size() >= batch_size:
                    # batch update

                    s1_batch, s1w_batch, a1_batch, s1y_batch, rs_batch, \
                        t_batch, sf_batch, sfw_batch = self.buffer.sample_batch(batch_size)

                    # Calculate targets
                    target_q = self.critic.predict_target(inputs=sf_batch, 
                                                          action=self.actor.predict_target(inputs=sf_batch,
                                                                                           portfolio_inputs=sfw_batch),
                                                          portfolio_inputs=sfw_batch)

                    y_i = []
                    for k in range(batch_size):
                        total_r = 0
                        for r_batch in reversed(rs_batch):
                            total_r *= gamma
                            total_r += r_batch[k]
                        if t_batch[k]:
                            y_i.append(total_r)
                        else:
                            y_i.append(total_r + (gamma**len(rs_batch))*target_q[k])

                    # Update the critic given the targets
                    predicted_q_value, _ = self.critic.train(inputs=s1_batch, 
                                                             action=a1_batch, 
                                                             predicted_q_value=np.reshape(y_i, (batch_size, 1)),
                                                             portfolio_inputs=s1w_batch,
                                                             future_y_inputs=s1y_batch)

                    ep_ave_max_q += np.amax(predicted_q_value)
                    ep_ave_min_q += np.amin(predicted_q_value)

                    # Update the actor policy using the sampled gradient
                    a_outs = self.actor.predict(inputs=s1_batch,
                                                portfolio_inputs=s1w_batch)
                    grads = self.critic.action_gradients(inputs=s1_batch, 
                                                         actions=a_outs,
                                                         portfolio_inputs=s1w_batch)
                    self.actor.train(inputs=s1_batch, 
                                     a_gradient=grads[0],
                                     portfolio_inputs=s1w_batch,
                                     future_y_inputs=s1y_batch)

                    # Update target networks
                    self.actor.update_target_network()
                    self.critic.update_target_network()

                ep_reward += reward
                [episode_rollout.popleft() for _ in range(5)]

                if done or j == self.max_rollout_steps - 1:
                    summary_str = self.sess.run(self.summary_ops, feed_dict={
                        self.summary_vars[0]: ep_reward,
                        self.summary_vars[1]: ep_ave_max_q / float(j)
                    })

                    writer.add_summary(summary_str, i)
                    writer.flush()

                    if (i % 10) == 0:
                        print("INFERRING")
                        self.infer(i, True)
                        self.infer(i, False)

                    if ((i+1) % 50) == 0:
                        print("SAVING")
                        self.save_model(i, 7, verbose=True)

                    print('Episode: {:d}, Reward: {:.2f}, Qmax: {:.4f}, Qmin{:.4f}'.format(i, 
                        ep_reward, (ep_ave_max_q / float(j)), (ep_ave_min_q / float(j))))
                    break

        self.save_model(i, 7, verbose=True)
        print('Finish.')

    # def train(self, save_every_episode=1, verbose=True, debug=False):
    #     """ Must already call intialize

    #     Args:
    #         save_every_episode:
    #         print_every_step:
    #         verbose:
    #         debug:

    #     Returns:

    #     """
    #     writer = tf.summary.FileWriter(self.summary_path, self.sess.graph)

    #     self.actor.update_target_network()
    #     self.critic.update_target_network()

    #     np.random.seed(self.config['seed'])
    #     num_episode = self.config['episode']
    #     batch_size = self.config['batch size']
    #     gamma = self.config['gamma']
    #     self.buffer = ReplayBufferMultiple(self.config['buffer size'])

        # # main training loop
        # for i in range(num_episode):
        #     if verbose and debug:
        #         print("Episode: " + str(i) + " Replay Buffer " + str(self.buffer.count()))

        #     observation_1 = self.env.reset()
        #     observation_1, weights_1 = observation_1[0]['obs'], observation_1[0]['weights']

        #     if self.obs_normalizer:
        #         observation_1 = self.obs_normalizer(observation_1)

        #     action_1 = self.actor.predict(inputs=np.expand_dims(observation_1, axis=0),
        #                                   portfolio_inputs=np.expand_dims(weights_1, axis=0)).squeeze(
        #                                   axis=0) + self.actor_noise()
        #     action_1 = np.clip(action_1, 0, 1)
        #     if action_1.sum() == 0:
        #         action_1 = np.ones(observation_1.shape[0])/observation_1.shape[0]
        #     action_1 /= action_1.sum()

        #     observation_2, reward_1, done, _ = self.env.step(action_1)
        #     observation_2, weights_2 = observation_2['obs'], observation_2['weights']

        #     if self.obs_normalizer:
        #         observation_2 = self.obs_normalizer(observation_2)

        #     ep_reward = 0
        #     ep_ave_max_q = 0
        #     ep_ave_min_q = 0
        #     # keeps sampling until done
        #     for j in range(self.config['max step']):
        #         action_2 = self.actor.predict(inputs=np.expand_dims(observation_2, axis=0),
        #                                       portfolio_inputs=np.expand_dims(weights_2, axis=0)).squeeze(
        #             axis=0) + self.actor_noise()

        #         if self.action_processor:
        #             action_2 = self.action_processor(action_2)
        #         else:
        #             action_2 = action_2
        #         # step forward

        #         #print("ACTION:", action_take)
        #         action_2 = np.clip(action_2, 0, 1)
        #         if action_2.sum() == 0:
        #             action_2 = np.ones(observation_1.shape[0])/observation_1.shape[0]
        #         action_2 /= action_2.sum()
        #         observation_3, reward_2, done, _ = self.env.step(action_2)
        #         observation_3, weights_3 = observation_3['obs'], observation_3['weights']

        #         if self.obs_normalizer:
        #             observation_3 = self.obs_normalizer(observation_3)

        #         # add to buffer
        #         self.buffer.add([observation_1, weights_1], action_1, reward_1, 
        #                         [observation_2, weights_2], action_2, reward_2,
        #                         done,
        #                         [observation_3, weights_3])

        #         if self.buffer.size() >= batch_size:
        #             # batch update
        #             s1_batch, s1w_batch, a1_batch, r1_batch, \
        #                 s2_batch, s2w_batch, a2_batch, r2_batch, \
        #                 t_batch, s3_batch, s3w_batch = self.buffer.sample_batch(batch_size)

        #             # Calculate targets
        #             target_q = self.critic.predict_target(inputs=s3_batch, 
        #                                                   action=self.actor.predict_target(inputs=s3_batch,
        #                                                                                    portfolio_inputs=s3w_batch),
        #                                                   portfolio_inputs=s3w_batch)

        #             y_i = []
        #             for k in range(batch_size):
        #                 if t_batch[k]:
        #                     y_i.append(r1_batch[k] + gamma * r2_batch[k])
        #                 else:
        #                     y_i.append(r1_batch[k] + gamma * r2_batch[k] + (gamma**2)*target_q[k])

        #             # Update the critic given the targets
        #             predicted_q_value, _ = self.critic.train(inputs=s1_batch, 
        #                                                      action=a1_batch, 
        #                                                      predicted_q_value=np.reshape(y_i, (batch_size, 1)),
        #                                                      portfolio_inputs=s1w_batch)

        #             ep_ave_max_q += np.amax(predicted_q_value)
        #             ep_ave_min_q += np.amin(predicted_q_value)

        #             # Update the actor policy using the sampled gradient
        #             a_outs = self.actor.predict(inputs=s1_batch,
        #                                         portfolio_inputs=s1w_batch)
        #             grads = self.critic.action_gradients(inputs=s1_batch, 
        #                                                  actions=a_outs,
        #                                                  portfolio_inputs=s1w_batch)
        #             self.actor.train(inputs=s1_batch, 
        #                              a_gradient=grads[0],
        #                              portfolio_inputs=s1w_batch)

        #             # Update target networks
        #             self.actor.update_target_network()
        #             self.critic.update_target_network()

        #         ep_reward += reward_1
        #         observation_1 = observation_2
        #         weights_1 = weights_2
        #         reward_1 = reward_2
        #         action_1 = action_2
        #         observation_2 = observation_3
        #         weights_2 = weights_3

        #         if done or j == self.config['max step'] - 1:
        #             summary_str = self.sess.run(self.summary_ops, feed_dict={
        #                 self.summary_vars[0]: ep_reward,
        #                 self.summary_vars[1]: ep_ave_max_q / float(j)
        #             })

        #             writer.add_summary(summary_str, i)
        #             writer.flush()

        #             if (i % 10) == 0:
        #                 self.infer(i, True)
        #                 self.infer(i, False)
        #             print('Episode: {:d}, Reward: {:.2f}, Qmax: {:.4f}, Qmin{:.4f}'.format(i, 
        #                 ep_reward, (ep_ave_max_q / float(j)), (ep_ave_min_q / float(j))))
        #             break

        # self.save_model(verbose=True)
        # print('Finish.')

    def predict(self, observation):
        """ predict the next action using actor model, only used in deploy.
            Can be used in multiple environments.

        Args:
            observation: (batch_size, num_stocks + 1, window_length)

        Returns: action array with shape (batch_size, num_stocks + 1)

        """
        if self.obs_normalizer:
            observation = self.obs_normalizer(observation)
        action = self.actor.predict(observation)
        if self.action_processor:
            action = self.action_processor(action)
        return action

    def predict_single(self, observation):
        """ Predict the action of a single observation

        Args:
            observation: (num_stocks + 1, window_length)

        Returns: a single action array with shape (num_stocks + 1,)

        """
        if self.obs_normalizer:
            observation = self.obs_normalizer(observation)
        action = self.actor.predict(np.expand_dims(observation, axis=0)).squeeze(axis=0)
        if self.action_processor:
            action = self.action_processor(action)
        return action

    def save_model(self, episode, max_to_keep=5, verbose=False):
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path, exist_ok=True)

        saver = tf.train.Saver(max_to_keep=max_to_keep)
        model_path = saver.save(self.sess, os.path.join(self.model_save_path, "checkpoint.ckpt"), 
                                global_step=episode)
        print("Model saved in %s" % model_path)

    # def infer(self, episode, train):
    #     if not train:
    #         env = self.env
    #     else:
    #         env = self.test_env

    #     observation_1 = env.reset()
    #     observation_1, weights_1 = observation_1[0]['obs'], observation_1[0]['weights']

    #     if self.obs_normalizer:
    #         observation_1 = self.obs_normalizer(observation_1)

    #     action_1 = self.actor.predict_target(inputs=np.expand_dims(observation_1, axis=0),
    #                                          portfolio_inputs=np.expand_dims(weights_1, axis=0)).squeeze(
    #                                          axis=0)
    #     action_1 = np.clip(action_1, 0, 1)
    #     if action_1.sum() == 0:
    #         action_1 = np.ones(observation_1.shape[0])/observation_1.shape[0]
    #     action_1 /= action_1.sum()

    #     observation_2, reward_1, done, _ = env.step(action_1)
    #     observation_2, weights_2 = observation_2['obs'], observation_2['weights']

    #     if self.obs_normalizer:
    #         observation_2 = self.obs_normalizer(observation_2)

    #     for j in range(env.steps-1):
    #         action_2 = self.actor.predict_target(inputs=np.expand_dims(observation_2, axis=0),
    #                                              portfolio_inputs=np.expand_dims(weights_2, axis=0)).squeeze(
    #                                              axis=0)

    #         if self.action_processor:
    #             action_2 = self.action_processor(action_2)
    #         else:
    #             action_2 = action_2
    #         # step forward

    #         #print("ACTION:", action_take)
    #         action_2 = np.clip(action_2, 0, 1)
    #         if action_2.sum() == 0:
    #             action_2 = np.ones(observation_1.shape[0])/observation_1.shape[0]
    #         action_2 /= action_2.sum()
    #         observation_3, reward_2, done, _ = env.step(action_2)
    #         observation_3, weights_3 = observation_3['obs'], observation_3['weights']

    #         if self.obs_normalizer:
    #             observation_3 = self.obs_normalizer(observation_3)

    #         observation_1 = observation_2
    #         weights_1 = weights_2
    #         reward_1 = reward_2
    #         action_1 = action_2
    #         observation_2 = observation_3
    #         weights_2 = weights_3

    #         if done or j == env.steps-2:
    #             label = 'train' if train else 'test'
    #             env.render()
    #             plt.savefig(os.path.join('./infer_' + label + '/', str(episode)+".png"))
    #             plt.close()
    #             break


    def infer(self, episode, train):
        """ Must already call intialize

        Args:
            save_every_episode:
            print_every_step:
            verbose:
            debug:

        Returns:

        """
        if not train:
            env = self.infer_test_env
        else:
            env = self.infer_train_env

        episode_rollout = deque()

        observation_1 = env.reset()
        observation_1, weights_1 = observation_1[0]['obs'], observation_1[0]['weights']

        if self.obs_normalizer:
            observation_1 = self.obs_normalizer(observation_1)

        episode_rollout.append([observation_1, weights_1])

        for rollout_step in range(self.learning_steps - 1):
            obs, ws = episode_rollout[-1]
            action = self.actor.predict(inputs=np.expand_dims(obs, axis=0),
                                        portfolio_inputs=np.expand_dims(ws, axis=0)).squeeze(
                                        axis=0)
            action = np.clip(action, 0, 1)
            if action.sum() == 0:
                action = np.ones(obs.shape[0])/obs.shape[0]
            action /= action.sum()
            new_obs, reward, done, _ = env.step(action)
            new_obs, new_ws = new_obs['obs'], new_obs['weights']

            if self.obs_normalizer:
                new_obs = self.obs_normalizer(new_obs)
            episode_rollout.append(action)
            episode_rollout.append(reward)
            episode_rollout.append(done)
            episode_rollout.append([new_obs, new_ws])

        for j in range(env.steps-self.learning_steps):
            action = self.actor.predict(inputs=np.expand_dims(episode_rollout[-1][0], axis=0),
                                        portfolio_inputs=np.expand_dims(episode_rollout[-1][1], axis=0)).squeeze(
                                        axis=0)

            if self.action_processor:
                action = self.action_processor(action)
            else:
                action = action

            action = np.clip(action, 0, 1)
            if action.sum() == 0:
                action = np.ones(episode_rollout[-1][0].shape[0])/episode_rollout[-1][0].shape[0]
            action /= action.sum()

            obs, reward, done, _ = env.step(action)
            obs, ws = obs['obs'], obs['weights']

            if self.obs_normalizer:
                obs = self.obs_normalizer(obs)

            episode_rollout.append(action)
            episode_rollout.append(reward)
            episode_rollout.append(done)
            episode_rollout.append([obs, ws])

            [episode_rollout.popleft() for _ in range(4)]

            if done or j == env.steps-self.learning_steps-1:
                label = 'train' if train else 'test'
                env.render()
                plt.savefig(os.path.join(self.infer_path, label + '/', str(episode)+".png"))
                plt.close()
                break