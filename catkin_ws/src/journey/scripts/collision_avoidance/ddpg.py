#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Sean Kirmani <sean@kirmani.io>
#
# Distributed under terms of the MIT license.
"""
Deep deterministic policy gradients with hindsight experience replay.
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import sys
import traceback
import tensorflow as tf
import time
from collections import deque


class DeepDeterministicPolicyGradients:

    def __init__(self,
                 create_actor_network,
                 create_critic_network,
                 action_dim,
                 use_discrete_actions=False,
                 gamma=0.99,
                 horizon=16,
                 use_hindsight=False):
        self.action_dim = action_dim
        self.use_discrete_actions = use_discrete_actions
        self.gamma = gamma
        self.horizon = horizon
        self.use_hindsight = use_hindsight

        # Start tensorflow session.
        self.sess = tf.Session()

        # Create actor network.
        self.actor = ActorNetwork(self.sess, create_actor_network, horizon, 128)
        self.critic = CriticNetwork(self.sess, create_critic_network,
                                    self.horizon,
                                    self.actor.get_num_trainable_vars())

    def build_summaries(self):
        episode_reward = tf.Variable(0.)
        tf.summary.scalar("Reward", episode_reward)
        critic_loss = tf.Variable(0.)
        tf.summary.scalar("Critic_loss", critic_loss)
        expected_collision_reward = tf.Variable(0.)
        tf.summary.scalar("expected_collision_reward",
                          expected_collision_reward)
        expected_task_reward = tf.Variable(0.)
        tf.summary.scalar("expected_task_reward", expected_task_reward)
        collision_model_loss = tf.Variable(0.)
        tf.summary.scalar("collision_model_loss", collision_model_loss)
        task_model_loss = tf.Variable(0.)
        tf.summary.scalar("task_model_loss", task_model_loss)

        summary_vars = [
            episode_reward, critic_loss, expected_collision_reward,
            expected_task_reward, collision_model_loss, task_model_loss
        ]
        summary_ops = tf.summary.merge_all()

        return summary_ops, summary_vars

    def eval(self, env, model_dir, num_attempts=1, max_episode_len=50):
        saver = tf.train.Saver()
        saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))

        for i in range(num_attempts):
            total_reward = 0.0
            state = env.Reset()
            action_horizon_idx = 0
            for j in range(max_episode_len):
                if action_horizon_idx == 0:
                    action_horizon = self.actor.predict(
                        np.expand_dims(state, axis=0))[0]
                action = action_horizon[action_horizon_idx]
                action_horizon_idx = (action_horizon_idx + 1) % self.horizon

                (next_state, action) = env.Step(state, action)
                terminal = env.Terminal(state, action)
                reward = env.Reward(next_state, action)

                state = next_state
                total_reward += reward[1]

                if terminal:
                    break

            print("Episode over.")
            print("Reward: %.4f" % total_reward)

    def train(self,
              env,
              actor_noise=None,
              act_randomly=False,
              k=1000,
              logdir='log',
              optimization_steps=40,
              num_epochs=200,
              episodes_in_epoch=16,
              max_episode_len=50,
              minibatch_size=128,
              model_dir=None):

        # Create a saver object for saving and loading variables
        saver = tf.train.Saver(max_to_keep=20)

        global_step = tf.Variable(0, trainable=False, name='global_step')
        increment_global_step = tf.assign_add(
            global_step, 1, name='increment_global_step')

        if model_dir != None:
            saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))
            print("Restoring model: %s" % model_dir)
            last_step = int(
                os.path.basename(
                    tf.train.latest_checkpoint(model_dir)).split('-')[1])

        # Set up summary Ops
        summary_ops, summary_vars = self.build_summaries()

        # Create a new log directory (if you run low on disk space you can
        # either disable this or delete old logs)
        # run: `tensorboard --logdir log` to see all the nice summaries
        for n_model in range(1000):
            summary_dir = "%s/model_%d" % (logdir, n_model)
            if not os.path.exists(summary_dir):
                break
        writer = tf.summary.FileWriter(summary_dir, self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        # Initialize target network weights
        self.actor.update_target_network()
        self.critic.update_target_network()

        # Initialize replay memory
        replay_buffer = ReplayBuffer()

        epsilon_zero = 0.2

        while tf.train.global_step(self.sess, global_step) < num_epochs:
            epoch = tf.train.global_step(self.sess, global_step)
            epoch_rewards = []
            total_epoch_avg_max_q = 0.0
            epsilon = epsilon_zero * (1.0 - epoch / num_epochs)
            for i in range(episodes_in_epoch):
                state = env.Reset()
                episode_reward = 0.0

                state_buffer = []
                action_buffer = []
                reward_buffer = []
                terminal_buffer = []
                next_state_buffer = []

                if actor_noise != None:
                    actor_noise.reset()

                for j in range(max_episode_len):
                    action_sequence = self.actor.predict(
                        np.expand_dims(state, axis=0))

                    action = action_sequence[0][0]

                    if self.use_discrete_actions:
                        # Epsilon-greedy exploration for discrete actions.
                        if np.random.random() < epsilon:
                            action = np.random.random(self.action_dim)
                            action = action / np.sum(action)
                    else:
                        # Added exploration noise.
                        if actor_noise != None:
                            action += actor_noise()

                    critique = self.critic.predict(
                        np.expand_dims(state, axis=0), action_sequence)[0]

                    (next_state, action) = env.Step(state, action, critique)
                    terminal = env.Terminal(next_state, action)
                    reward = env.Reward(next_state, action)

                    # Add to episode buffer.
                    state_buffer.append(state)
                    action_buffer.append(action)
                    reward_buffer.append(reward)
                    terminal_buffer.append(terminal)
                    next_state_buffer.append(next_state)

                    state = next_state
                    episode_reward += reward[1]

                    if terminal:
                        break

                for t in range(len(state_buffer)):
                    y_i = np.zeros((self.horizon, 2))
                    a_i = np.zeros((self.horizon, self.action_dim))
                    for h in range(self.horizon):
                        if t + h < len(state_buffer):
                            y_i[h, :] = np.array(reward_buffer[t + h])
                            a_i[h, :] = action_buffer[t + h]
                        else:
                            y_i[h, :] = np.array(reward_buffer[-1])
                            a_i[h, :] = np.random.random(self.action_dim)
                    replay_buffer.add(state_buffer[j], a_i, y_i,
                                      terminal_buffer[j], next_state_buffer[j])

                epoch_rewards.append(episode_reward)

                if np.isnan(episode_reward):
                    print("Reward is NaN. Exiting...")
                    sys.exit(0)

                print('| Reward: {:4f} | Episode: {:d} |'.format(
                    episode_reward, i))

            print("Finished data collection for epoch %d." % epoch)
            print("Experience buffer size: %s" % replay_buffer.size())
            if replay_buffer.size() < minibatch_size:
                continue

            print("Starting policy optimization.")
            for optimization_step in range(optimization_steps):
                batch_size = min(minibatch_size, replay_buffer.size())
                (s_batch, a_batch, r_batch, t_batch,
                 s2_batch) = replay_buffer.sample_batch(batch_size)

                # Calculate targets
                target_q = self.critic.predict_target(
                    s2_batch, self.actor.predict_target(s2_batch))
                target_q[:, :, 0] = 1.0 / (
                    1.0 + np.exp(-np.array(target_q[:, :, 0])))

                # Y represents the probability of flight without collision
                #   between time t and t + h.
                # B represents the best-case future likelihood of flight
                #   without collosion.
                y_coll_i = np.zeros((batch_size, self.horizon))
                b_coll_i = np.zeros(batch_size)
                y_task_i = np.zeros((batch_size, self.horizon))
                b_task_i = np.zeros(batch_size)
                for k in range(batch_size):
                    y_coll_i[k, :] = r_batch[k, :, 0]
                    b_coll_i[k] = np.mean(target_q[k, :self.horizon, 0])
                    y_task_i[k, :] = r_batch[k, :, 1]
                    b_task_i[k] = np.mean(target_q[k, :self.horizon, 1])

                # Update the critic given the targets
                (critic_loss, collision_model_loss, task_model_loss,
                 expected_collision_reward,
                 expected_task_reward) = self.critic.train(
                     s_batch,
                     a_batch,
                     np.reshape(y_coll_i, (batch_size, self.horizon)),
                     np.reshape(b_coll_i, (batch_size, 1)),
                     np.reshape(y_task_i, (batch_size, self.horizon)),
                     np.reshape(b_task_i, (batch_size, 1)),
                 )

                # Update the actor policy using the sampled gradient
                a_outs = self.actor.predict(s_batch)
                grads = self.critic.action_gradients(s_batch, a_outs)
                self.actor.train(s_batch, grads[0])

                # Update target networks
                self.actor.update_target_network()
                self.critic.update_target_network()

            # Output training statistics.
            print(
                "[%d] Critic Loss: %.4f, Exp Collision Reward: %.4f, Exp Task Reward: %.4f"
                % (optimization_step, critic_loss, expected_collision_reward,
                   expected_task_reward))

            average_epoch_reward = np.mean(epoch_rewards)
            epoch_reward_stddev = np.std(epoch_rewards)
            if np.isnan(average_epoch_reward):
                print("Reward is NaN. Exiting...")
                sys.exit(0)
            print('| Reward: {:4f} ({:4f})| Epoch: {:d} |'.format(
                average_epoch_reward, epoch_reward_stddev, epoch))

            # Write episode summary statistics.
            summary_str = self.sess.run(
                summary_ops,
                feed_dict={
                    summary_vars[0]: average_epoch_reward,
                    summary_vars[1]: critic_loss,
                    summary_vars[2]: expected_collision_reward,
                    summary_vars[3]: expected_task_reward,
                    summary_vars[4]: collision_model_loss,
                    summary_vars[5]: task_model_loss,
                })

            writer.add_summary(summary_str, epoch)
            writer.flush()

            # Save model checkpoint.
            print("Saving model checkpoint: %s/model.ckpt-%d.meta" %
                  (summary_dir, epoch))
            saver.save(
                self.sess, summary_dir + '/model.ckpt', global_step=epoch)
            self.sess.run(increment_global_step)


class Environment:

    def __init__(self, reset, step, reward, terminal):
        self.reset = reset
        self.step = step
        self.reward = reward
        self.terminal = terminal

    def Reset(self):
        return self.reset()

    def Step(self, state, action, critique):
        return self.step(state, action, critique)

    def Reward(self, state, action):
        return self.reward(state, action)

    def Terminal(self, state, action):
        return self.terminal(state, action)


class ReplayBuffer:

    def __init__(self, buffer_size=1000000):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        '''
        batch_size specifies the number of experiences to add
        to the batch. If the replay buffer has less than batch_size
        elements, simply return all of the elements within the buffer.
        Generally, you'll want to wait until the buffer has at least
        batch_size elements before beginning to sample from it.
        '''
        batch = []

        if self.count < batch_size:
            batch = random.sample(list(self.buffer), self.count)
        else:
            batch = random.sample(list(self.buffer), batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0


class ActorNetwork:

    def __init__(self,
                 sess,
                 create_actor_network,
                 horizon,
                 batch_size,
                 tau=0.001,
                 learning_rate=0.0001):
        self.sess = sess

        # Actor network.
        self.inputs, self.actions = create_actor_network("actor_source")
        network_params = tf.trainable_variables()
        print("Actor network has %s parameters." % np.sum(
            [v.get_shape().num_elements() for v in network_params]))

        # Target network.
        self.target_inputs, self.target_actions = create_actor_network(
            "actor_target")
        target_network_params = tf.trainable_variables()[len(network_params):]

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [target_network_params[i].assign(tf.multiply(network_params[i], tau) + \
                tf.multiply(target_network_params[i], 1. - tau))
                for i in range(len(target_network_params))]

        # This gradient will be provided by the critic network
        self.action_dim = self.actions.shape[-1]
        self.action_gradient = tf.placeholder(tf.float32,
                                              [None, horizon, self.action_dim])

        # Combine the gradients here
        unnormalized_actor_gradients = tf.gradients(
            self.actions, network_params, -self.action_gradient)
        self.actor_gradients = list(
            map(lambda x: tf.div(x, batch_size), unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(learning_rate).\
            apply_gradients(zip(self.actor_gradients, network_params))

        self.num_trainable_vars = len(network_params) + len(
            target_network_params)

    def train(self, inputs, a_gradient):
        self.sess.run(
            self.optimize,
            feed_dict={
                self.inputs: inputs,
                self.action_gradient: a_gradient
            })

    def predict(self, inputs):
        return self.sess.run(self.actions, feed_dict={self.inputs: inputs})

    def predict_target(self, inputs):
        return self.sess.run(
            self.target_actions, feed_dict={
                self.target_inputs: inputs
            })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork:

    def __init__(self,
                 sess,
                 create_critic_network,
                 horizon,
                 num_actor_vars,
                 collision_weight=0.01,
                 tau=0.001,
                 learning_rate=0.001):
        self.sess = sess

        # Critic network.
        (self.inputs, self.actions, self.y_coll_out, self.b_coll_out,
         self.y_task_out,
         self.b_task_out) = create_critic_network("critic_source")
        network_params = tf.trainable_variables()[num_actor_vars:]
        print("Critic network has %s parameters." % np.sum(
            [v.get_shape().num_elements() for v in network_params]))

        # Target network.
        (self.target_inputs, self.target_actions, self.target_y_coll_out,
         self.target_b_coll_out, self.target_y_task_out,
         self.target_b_task_out) = create_critic_network("critic_target")
        target_network_params = tf.trainable_variables()[(
            len(network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [target_network_params[i].assign(tf.multiply(network_params[i], tau) + \
                tf.multiply(target_network_params[i], 1. - tau))
                for i in range(len(target_network_params))]

        # Network target (y_i)
        # Obtained from the target networks
        self.predicted_y_coll_value = tf.placeholder(tf.float32,
                                                     (None, horizon))
        self.predicted_b_coll_value = tf.placeholder(tf.float32, (None, 1))
        self.predicted_y_task_value = tf.placeholder(tf.float32,
                                                     (None, horizon))
        self.predicted_b_task_value = tf.placeholder(tf.float32, (None, 1))

        # Define loss and optimization Op
        y_coll_loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.predicted_y_coll_value, logits=self.y_coll_out),
            axis=1)
        b_coll_loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.predicted_b_coll_value, logits=self.b_coll_out),
            axis=1)
        self.collision_loss = tf.reduce_mean(y_coll_loss + b_coll_loss)

        y_task_loss = tf.reduce_sum(
            (self.predicted_y_task_value - self.y_task_out)**2, axis=1)
        b_task_loss = tf.reduce_sum(
            (self.predicted_b_task_value - self.b_task_out)**2, axis=1)
        self.task_loss = tf.reduce_mean(y_task_loss + b_task_loss)

        self.loss = self.task_loss + collision_weight * self.collision_loss

        # self.loss = tf.reduce_mean((self.predicted_q_value - self.y_coll_out)**2)
        self.optimize = tf.train.AdamOptimizer(learning_rate).minimize(
            self.loss)

        # Metrics
        self.model_accuracy = 1.0 - tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.predicted_y_coll_value, logits=self.y_coll_out))
        self.expected_collision_reward = tf.reduce_mean(
            tf.nn.sigmoid(self.b_coll_out))
        self.expected_task_reward = tf.reduce_mean(self.b_task_out)

        # Get the gradient of the net w.r.t. the action
        task_influence = tf.reduce_sum(
            self.y_task_out, axis=1, keep_dims=True) + self.b_task_out
        collision_influence = tf.reduce_sum(
            self.y_coll_out, axis=1, keep_dims=True) + self.b_coll_out
        critic_influence = (
            task_influence + collision_weight * collision_influence)
        self.action_grads = tf.gradients(critic_influence, self.actions)

    def train(self, inputs, actions, y_coll, b_coll, y_task, b_task):
        (loss, collision_model_loss, task_model_loss, expected_collision_reward,
         expected_task_reward, _) = self.sess.run(
             [
                 self.loss, self.collision_loss, self.task_loss,
                 self.expected_collision_reward, self.expected_task_reward,
                 self.optimize
             ],
             feed_dict={
                 self.inputs: inputs,
                 self.actions: actions,
                 self.predicted_y_coll_value: y_coll,
                 self.predicted_b_coll_value: b_coll,
                 self.predicted_y_task_value: y_task,
                 self.predicted_b_task_value: b_task,
             })
        return (loss, collision_model_loss, task_model_loss,
                expected_collision_reward, expected_task_reward)

    def predict(self, inputs, actions):
        preds = self.sess.run(
            [
                self.y_coll_out, self.b_coll_out, self.y_task_out,
                self.b_task_out
            ],
            feed_dict={
                self.inputs: inputs,
                self.actions: actions
            })
        y_coll_out = np.array(preds[0])
        b_coll_out = np.array(preds[1])
        y_task_out = np.array(preds[2])
        b_task_out = np.array(preds[3])
        coll_preds = np.concatenate([y_coll_out, b_coll_out], axis=1)
        task_preds = np.concatenate([y_task_out, b_task_out], axis=1)
        preds = np.stack([coll_preds, task_preds], axis=-1)
        return preds

    def predict_target(self, inputs, actions):
        preds = self.sess.run(
            [
                self.target_y_coll_out, self.target_b_coll_out,
                self.target_y_task_out, self.target_b_task_out
            ],
            feed_dict={
                self.target_inputs: inputs,
                self.target_actions: actions
            })
        y_coll_out = np.array(preds[0])
        b_coll_out = np.array(preds[1])
        y_task_out = np.array(preds[2])
        b_task_out = np.array(preds[3])
        coll_preds = np.concatenate([y_coll_out, b_coll_out], axis=1)
        task_preds = np.concatenate([y_task_out, b_task_out], axis=1)
        preds = np.stack([coll_preds, task_preds], axis=-1)
        return preds

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def action_gradients(self, inputs, actions):
        return self.sess.run(
            self.action_grads,
            feed_dict={
                self.inputs: inputs,
                self.actions: actions
            })


class OrnsteinUhlenbeckActionNoise:

    def __init__(self, mu, sigma=0.1, theta=0.1, dt=0.25, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
            self.mu, self.sigma)
