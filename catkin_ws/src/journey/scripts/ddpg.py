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
                 num_inputs,
                 image_width,
                 image_height,
                 num_actions,
                 goal_dim,
                 minibatch_size=128,
                 gamma=0.98):
        self.num_inputs = num_inputs
        self.image_width = image_width
        self.image_height = image_height
        self.num_actions = num_actions
        self.goal_dim = goal_dim
        self.minibatch_size = minibatch_size
        self.gamma = gamma

        # Start tensorflow session.
        self.sess = tf.Session()

        # Create actor network.
        self.actor = ActorNetwork(self.sess, self.num_inputs, self.image_width,
                                  self.image_height, self.num_actions,
                                  self.goal_dim)
        self.critic = CriticNetwork(self.sess, self.num_inputs,
                                    self.image_width, self.image_height,
                                    self.num_actions, self.goal_dim,
                                    self.actor.get_num_trainable_vars())

        self.sess.run(tf.global_variables_initializer())

    def build_summaries(self):
        episode_reward = tf.Variable(0.)
        tf.summary.scalar("Reward", episode_reward)
        episode_ave_max_q = tf.Variable(0.)
        tf.summary.scalar("Qmax_Value", episode_ave_max_q)

        summary_vars = [episode_reward, episode_ave_max_q]
        summary_ops = tf.summary.merge_all()

        return summary_ops, summary_vars

    def Train(self,
              env,
              actor_noise,
              logdir='log',
              optimization_steps=40,
              num_epochs=200,
              episodes_in_epoch=16,
              max_episode_len=50):
        # Set up summary Ops
        summary_ops, summary_vars = self.build_summaries()

        # Create a new log directory (if you run low on disk space you can either disable this or delete old logs)
        # run: `tensorboard --logdir log` to see all the nice summaries
        for n_model in range(1000):
            summary_dir = "%s/model_%d" % (logdir, n_model)
            if not os.path.exists(summary_dir):
                break
        writer = tf.summary.FileWriter(summary_dir, self.sess.graph)

        # Initialize target network weights
        self.actor.update_target_network()
        self.critic.update_target_network()

        # Initialize replay memory
        replay_buffer = ReplayBuffer()

        for epoch in range(num_epochs):
            total_epoch_reward = 0.0
            total_epoch_avg_max_q = 0.0
            for i in range(episodes_in_epoch):
                (state, goal) = env.Reset()
                episode_reward = 0.0

                state_buffer = []
                action_buffer = []
                reward_buffer = []
                terminal_buffer = []
                next_state_buffer = []

                for j in range(max_episode_len):
                    # Added exploration noise.
                    action = self.actor.predict(
                        np.expand_dims(
                            np.expand_dims(
                                np.concatenate([state, goal], axis=-1), axis=0),
                            axis=0))[0][0] + actor_noise()

                    next_state = env.Step(state, action, goal)
                    terminal = False
                    reward = env.Reward(next_state, action, goal)

                    # Add to episode buffer.
                    state_buffer.append(np.concatenate([state, goal], axis=-1))
                    action_buffer.append(action)
                    reward_buffer.append(reward)
                    terminal_buffer.append(terminal)
                    next_state_buffer.append(
                        np.concatenate([next_state, goal], axis=-1))

                    state = next_state
                    episode_reward += reward

                replay_buffer.add(state_buffer, action_buffer, reward_buffer,
                                  terminal_buffer, next_state_buffer)

                # Hindsight experience replay.
                for j in range(max_episode_len):
                    goal = next_state_buffer[j][:self.num_inputs]
                    her_state_buffer = []
                    her_action_buffer = []
                    her_reward_buffer = []
                    her_terminal_buffer = []
                    her_next_state_buffer = []
                    for k in range(max_episode_len):
                        # Simulate step with hindsight goal.
                        state = state_buffer[k][:self.num_inputs]
                        action = action_buffer[k]
                        next_state = next_state_buffer[k][:self.num_inputs]
                        terminal = False
                        reward = env.Reward(next_state, action, goal)

                        # Add to hindersight buffers.
                        her_state_buffer.append(
                            np.concatenate([state, goal], axis=-1))
                        her_action_buffer.append(action)
                        her_reward_buffer.append(reward)
                        her_terminal_buffer.append(terminal)
                        her_next_state_buffer.append(
                            np.concatenate([next_state, goal], axis=-1))

                    replay_buffer.add(her_state_buffer, her_action_buffer,
                                      her_reward_buffer, her_terminal_buffer,
                                      her_next_state_buffer)

                predicted_q_values = self.critic.predict(
                    np.expand_dims(state_buffer, axis=0),
                    np.expand_dims(action_buffer, axis=0))
                episode_max_q = np.amax(predicted_q_values)
                episode_avg_max_q = episode_max_q / max_episode_len
                total_epoch_reward += episode_reward
                total_epoch_avg_max_q += episode_avg_max_q
                average_epoch_reward = total_epoch_reward / (i + 1)
                average_epoch_avg_max_q = total_epoch_avg_max_q / (i + 1)

                print('| Reward: {:4f} | Episode: {:d} | Qmax: {:.4f} |'.format(
                    episode_reward, i, episode_avg_max_q))

            print("Finished data collection for epoch %d." % epoch)
            print("Training minibatch with %s trajectories." %
                  replay_buffer.size())
            print("Starting policy optimization.")
            average_epoch_avg_max_q = 0.0
            for optimization_step in range(optimization_steps):
                (s_batch, a_batch, r_batch, t_batch,
                 s2_batch) = replay_buffer.sample_batch(self.minibatch_size)

                # Calculate targets
                target_q = self.critic.predict_target(
                    s2_batch, self.actor.predict_target(s2_batch))

                y_i = []
                for k in range(self.minibatch_size):
                    for l in range(max_episode_len):
                        if t_batch[k][l]:
                            y_i.append(r_batch[k][l])
                        else:
                            y_i.append(r_batch[k][l] +
                                       self.gamma * target_q[k][l])

                # Update the critic given the targets
                predicted_q_value = self.critic.train(
                    s_batch, a_batch,
                    np.reshape(y_i, (self.minibatch_size, max_episode_len, 1)))
                average_epoch_avg_max_q += np.amax(predicted_q_value)
                print("[%d] Qmax: %.4f" %
                      (optimization_step,
                       average_epoch_avg_max_q / (optimization_step + 1)))

                # Update the actor policy using the sampled gradient
                a_outs = self.actor.predict(s_batch)
                grads = self.critic.action_gradients(s_batch, a_outs)
                self.actor.train(s_batch, grads[0])

                # Update target networks
                self.actor.update_target_network()
                self.critic.update_target_network()
            average_epoch_avg_max_q /= optimization_steps
            print('| Reward: {:4f} | Epoch: {:d} | Qmax: {:4f} |'.format(
                average_epoch_reward, epoch, average_epoch_avg_max_q))

            # Write episode summary statistics.
            summary_str = self.sess.run(
                summary_ops,
                feed_dict={
                    summary_vars[0]: average_epoch_reward,
                    summary_vars[1]: average_epoch_avg_max_q
                })

            writer.add_summary(summary_str, epoch)
            writer.flush()


class Environment:

    def __init__(self, reset, step, reward):
        self.reset = reset
        self.step = step
        self.reward = reward

    def Reset(self):
        return self.reset()

    def Step(self, state, action, goal):
        return self.step(state, action, goal)

    def Reward(self, state, action, goal):
        return self.reward(state, action, goal)


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
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

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
                 state_dim,
                 image_width,
                 image_height,
                 action_dim,
                 goal_dim,
                 tau=0.05,
                 learning_rate=0.0001):
        self.sess = sess
        self.state_dim = state_dim
        self.image_width = image_width
        self.image_height = image_height
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.num_inputs = (self.image_width * self.image_height + self.state_dim
                           + self.goal_dim)

        # Actor network.
        self.inputs, self.actions = self.create_actor_network("actor_source")
        network_params = tf.trainable_variables()

        # Target network.
        self.target_inputs, self.target_actions = self.create_actor_network(
            "actor_target")
        target_network_params = tf.trainable_variables()[len(network_params):]

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [target_network_params[i].assign(tf.multiply(network_params[i], tau) + \
                tf.multiply(target_network_params[i], 1. - tau))
                for i in range(len(target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32,
                                              [None, None, self.action_dim])

        # Combine the gradients here
        self.actor_gradients = tf.gradients(self.actions, network_params,
                                            -self.action_gradient)

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(learning_rate).\
            apply_gradients(zip(self.actor_gradients, network_params))

        self.num_trainable_vars = len(network_params) + len(
            target_network_params)

    def create_actor_network(self, scope):
        inputs = tf.placeholder(tf.float32, (None, None, self.num_inputs))
        position = inputs[:, :, :self.state_dim]
        depth = tf.reshape(inputs[:, :, self.state_dim:],
                           [-1, self.image_height, self.image_width, 1])
        depth = tf.image.resize_images(depth, [84, 84])
        depth = tf.contrib.layers.conv2d(
            depth,
            num_outputs=16,
            activation_fn=tf.nn.relu,
            kernel_size=(8, 8),
            stride=(4, 4),
            weights_regularizer=tf.nn.l2_loss)
        depth = tf.contrib.layers.conv2d(
            depth,
            num_outputs=32,
            activation_fn=tf.nn.relu,
            kernel_size=(5, 5),
            stride=(2, 2),
            weights_regularizer=tf.nn.l2_loss)
        depth = tf.contrib.layers.flatten(depth)
        depth = tf.contrib.layers.fully_connected(depth, 64)
        depth = tf.reshape(depth,
                           [tf.shape(position)[0],
                            tf.shape(position)[1], 64])
        position = tf.contrib.layers.fully_connected(position, 64)
        x = tf.concat([position, depth], axis=-1)
        cell = tf.contrib.rnn.LSTMCell(128)
        lstm_outputs, state = tf.nn.dynamic_rnn(
            cell, x, dtype=tf.float32, scope=scope)
        actions = tf.contrib.layers.fully_connected(
            inputs=lstm_outputs,
            num_outputs=self.action_dim,
            activation_fn=tf.tanh)
        return inputs, actions

    def train(self, inputs, a_gradient):
        self.sess.run(
            self.optimize,
            feed_dict={self.inputs: inputs,
                       self.action_gradient: a_gradient})

    def predict(self, inputs):
        preds = self.sess.run(self.actions, feed_dict={self.inputs: inputs})
        preds = np.reshape(preds,
                           [inputs.shape[0], inputs.shape[1], self.action_dim])
        return preds

    def predict_target(self, inputs):
        preds = self.sess.run(
            self.target_actions, feed_dict={self.target_inputs: inputs})
        preds = np.reshape(preds,
                           [inputs.shape[0], inputs.shape[1], self.action_dim])
        return preds

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork:

    def __init__(self,
                 sess,
                 state_dim,
                 image_width,
                 image_height,
                 action_dim,
                 goal_dim,
                 num_actor_vars,
                 tau=0.05,
                 learning_rate=0.001):
        self.sess = sess
        self.state_dim = state_dim
        self.image_width = image_width
        self.image_height = image_height
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.num_inputs = (self.image_width * self.image_height + self.state_dim
                           + self.goal_dim)

        # Critic network.
        (self.inputs, self.actions,
         self.out) = self.create_critic_network("critic_source")
        network_params = tf.trainable_variables()[num_actor_vars:]

        # Target network.
        (self.target_inputs, self.target_actions,
         self.target_out) = self.create_critic_network("critic_target")
        target_network_params = tf.trainable_variables()[(
            len(network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [target_network_params[i].assign(tf.multiply(network_params[i], tau) + \
                tf.multiply(target_network_params[i], 1. - tau))
                for i in range(len(target_network_params))]

        # Network target (y_i)
        # Obtained from the target networks
        self.predicted_q_value = tf.placeholder(tf.float32, (None, None, 1))

        # Define loss and optimization Op
        shaped_labels = tf.reshape(self.predicted_q_value, [-1, 1])
        self.loss = tf.reduce_mean((self.out - shaped_labels)**2)
        self.optimize = tf.train.AdamOptimizer(learning_rate).minimize(
            self.loss)

        # Get the gradient of the net w.r.t. the action
        shaped_out = tf.reshape(
            self.out, [tf.shape(self.inputs)[0],
                       tf.shape(self.inputs)[1], 1])
        self.action_grads = tf.gradients(shaped_out, self.actions)

    def create_critic_network(self, scope):
        inputs = tf.placeholder(tf.float32, (None, None, self.num_inputs))
        actions = tf.placeholder(tf.float32, (None, None, self.action_dim))
        position = inputs[:, :, :self.state_dim]
        depth = tf.reshape(inputs[:, :, self.state_dim:],
                           [-1, self.image_height, self.image_width, 1])
        depth = tf.image.resize_images(depth, [84, 84])
        depth = tf.contrib.layers.conv2d(
            depth,
            num_outputs=16,
            activation_fn=tf.nn.relu,
            kernel_size=(8, 8),
            stride=(4, 4),
            weights_regularizer=tf.nn.l2_loss)
        depth = tf.contrib.layers.conv2d(
            depth,
            num_outputs=32,
            activation_fn=tf.nn.relu,
            kernel_size=(5, 5),
            stride=(2, 2),
            weights_regularizer=tf.nn.l2_loss)
        depth = tf.contrib.layers.flatten(depth)
        depth = tf.contrib.layers.fully_connected(depth, 64)
        depth = tf.reshape(depth,
                           [tf.shape(position)[0],
                            tf.shape(position)[1], 64])
        position = tf.contrib.layers.fully_connected(
            tf.concat([position, actions], axis=-1), 64)
        x = tf.concat([position, depth], axis=-1)
        cell = tf.contrib.rnn.LSTMCell(128)
        lstm_outputs, state = tf.nn.dynamic_rnn(
            cell, x, dtype=tf.float32, scope=scope)
        out = tf.contrib.layers.fully_connected(
            inputs=lstm_outputs, num_outputs=1, activation_fn=None)
        return inputs, actions, out

    def train(self, inputs, actions, reward):
        preds, _ = self.sess.run(
            [self.out, self.optimize],
            feed_dict={
                self.inputs: inputs,
                self.actions: actions,
                self.predicted_q_value: reward
            })
        return preds

    def predict(self, inputs, actions):
        preds = self.sess.run(
            self.out, feed_dict={self.inputs: inputs,
                                 self.actions: actions})
        preds = np.reshape(preds, [inputs.shape[0], inputs.shape[1], 1])
        return preds

    def predict_target(self, inputs, actions):
        preds = self.sess.run(
            self.target_out,
            feed_dict={
                self.target_inputs: inputs,
                self.target_actions: actions
            })
        preds = np.reshape(preds, [inputs.shape[0], inputs.shape[1], 1])
        return preds

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def action_gradients(self, inputs, actions):
        return self.sess.run(
            self.action_grads,
            feed_dict={self.inputs: inputs,
                       self.actions: actions})


class OrnsteinUhlenbeckActionNoise:

    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
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


def main(args):
    """ Main function. """
    # TODO(kirmani): Do something more interesting here...
    print('Hello world!')


if __name__ == '__main__':
    try:
        start_time = time.time()
        parser = argparse.ArgumentParser(usage=globals()['__doc__'])
        parser.add_argument(
            '-v',
            '--verbose',
            action='store_true',
            default=False,
            help='verbose output')
        args = parser.parse_args()
        #if len(args) < 1:
        #    parser.error ('missing argument')
        if args.verbose:
            print(time.asctime())
        main(args)
        if args.verbose:
            print(time.asctime())
            print('TOTAL TIME IN MINUTES:',)
            print((time.time() - start_time) / 60.0)
        sys.exit(0)
    except KeyboardInterrupt as err:  # Ctrl-C
        raise err
    except SystemExit as err:  # sys.exit()
        raise err
    except Exception as err:
        print('ERROR, UNEXPECTED EXCEPTION')
        print(str(err))
        traceback.print_exc()
        sys.exit(1)
