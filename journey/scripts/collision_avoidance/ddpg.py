#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Sean Kirmani <sean@kirmani.io>
#
# Distributed under terms of the MIT license.
"""
Actor-critic generalized computation graph (AC-GCG). A model-based reinforcement
learning approach for navigation and obstacle avoidance.

Our critic is similar to N-step Q-Learning where we learn a Q-function, as well
as a model of the world.

In this implementation, we specifically are interested in learning
collision-avoidant navigation policies, so we simultaneously learn a model that
predicts the probability of collision as well as the model of our reward for
some arbitray navigation task.

Our actor network takes actions that maximize our expected return for our
navigation task weighted with our desire to avoid obstacles.

We can formulate the process as a 3-level optimization problem. We first are
trying to model the environment, then our critic is trying to evaluate our
environment given the learned model, and our actor is trying to optimize
actions with respect to our critic's evaluation of our model.
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import traceback
import tensorflow as tf
from environment import Environment
from replay_buffer import ReplayBuffer


class DeepDeterministicPolicyGradients:

    def __init__(self,
                 create_actor_network,
                 create_critic_network,
                 minibatch_size=128,
                 gamma=0.99,
                 horizon=16,
                 collision_weight=0.01,
                 discrete_controls=False):
        self.collision_weight = collision_weight
        self.gamma = gamma
        self.horizon = horizon
        self.minibatch_size = minibatch_size
        self.discrete_controls = discrete_controls

        # Start tensorflow session.
        self.sess = tf.Session()

        # Create actor and critic networks.
        self.actor = ActorNetwork(self.sess, create_actor_network, horizon,
                                  self.minibatch_size)
        self.critic = CriticNetwork(
            self.sess,
            create_critic_network,
            self.horizon,
            self.actor.get_num_trainable_vars(),
            collision_weight=self.collision_weight)
        self.action_dim = self.actor.get_action_dim()

    def build_summaries(self):
        episode_reward = tf.Variable(0.)
        tf.summary.scalar("reward", episode_reward)
        loss = tf.Variable(0.)
        tf.summary.scalar("loss", loss)
        expected_reward = tf.Variable(0.)
        tf.summary.scalar("expected_reward", expected_reward)

        summary_vars = [
            episode_reward,
            loss,
            expected_reward,
        ]
        summary_ops = tf.summary.merge_all()

        return summary_ops, summary_vars

    def eval(self, env, model_dir, num_attempts=1, max_episode_len=1000):
        saver = tf.train.Saver()
        saver.restore(self.sess, model_dir)

        for i in range(num_attempts):
            episode_reward = 0.0
            state = env.reset()
            for j in range(max_episode_len):
                # Predict the optimal actions over the horizon.
                action_sequence = self.actor.predict(
                    np.expand_dims(state, axis=0))

                # MPC action selection.
                action = action_sequence[0][0]

                # Take a step.
                next_state = env.step(state, action)
                terminal = env.terminal(next_state, action)
                reward = env.reward(next_state, action)

                state = next_state
                episode_reward += reward[0]

                if terminal:
                    break

            print("Episode over.")
            print("Reward: %.4f" % episode_reward)

    def load_model(self, model_dir):
        saver = tf.train.Saver()
        saver.restore(self.sess, model_dir)

    def train(self,
              env,
              actor_noise=None,
              logdir='log',
              optimization_steps=40,
              num_epochs=1000,
              episodes_in_epoch=16,
              max_episode_len=1000,
              epsilon_zero=0.2,
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

        if actor_noise is None:
            actor_noise = OrnsteinUhlenbeckActionNoise(
                mu=np.zeros(self.action_dim))

        while tf.train.global_step(self.sess, global_step) < num_epochs:
            epoch = tf.train.global_step(self.sess, global_step)
            epoch_rewards = []
            total_epoch_avg_max_q = 0.0

            epsilon = epsilon_zero * (1.0 - epoch / num_epochs)
            print("Explore with epsilon greedy (epsilon = %.4f)" % epsilon)

            for i in range(episodes_in_epoch):
                state = env.reset()
                episode_reward = 0.0

                state_buffer = []
                action_buffer = []
                reward_buffer = []
                terminal_buffer = []
                next_state_buffer = []
                actor_noise.reset()

                for j in range(max_episode_len):
                    # Predict the optimal actions over the horizon.
                    action_sequence = self.actor.predict(
                        np.expand_dims(state, axis=0))

                    # MPC action selection.
                    action = action_sequence[0][0]

                    # Added exploration noise.
                    if self.discrete_controls:
                        if np.random.random() < epsilon:
                            action = np.random.random(self.action_dim)
                    else:
                        action += actor_noise()

                    # Bound action within [-1, 1]
                    action = np.clip(action, -1, 1)

                    # Take a step.
                    next_state = env.step(state, action)
                    terminal = env.terminal(next_state, action)
                    reward = env.reward(next_state, action)

                    # DEBUG.
                    # self.critic.predict(
                    #     np.expand_dims(state, axis=0), action_sequence)
                    # env.visualize(state, action_sequence[0])

                    # Add to episode buffer.
                    state_buffer.append(state)
                    action_buffer.append(action)
                    reward_buffer.append(reward)
                    terminal_buffer.append(terminal)
                    next_state_buffer.append(next_state)

                    state = next_state
                    episode_reward += reward[0]

                    if terminal:
                        break

                # For our entire episode add our experience with our target
                # model rewards and future actions to our experience replay buffer.
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
                    replay_buffer.add(state_buffer[t], a_i, y_i,
                                      terminal_buffer[t], next_state_buffer[t])

                epoch_rewards.append(episode_reward)
                print('| Reward: {:4f} | Episode: {:d} |'.format(
                    episode_reward, i))

            print("Experience buffer size: %s" % replay_buffer.size())
            if replay_buffer.size() < self.minibatch_size:
                continue

            # Output epoch statistics.
            average_epoch_reward = np.mean(epoch_rewards)
            epoch_reward_stddev = np.std(epoch_rewards)
            print('| Reward: {:4f} ({:4f})| Epoch: {:d} |'.format(
                average_epoch_reward, epoch_reward_stddev, epoch))

            print("Finished data collection for epoch %d." % epoch)
            print("Starting policy optimization.")
            for optimization_step in range(optimization_steps):
                batch_size = min(self.minibatch_size, replay_buffer.size())
                (s_batch, a_batch, r_batch, t_batch,
                 s2_batch) = replay_buffer.sample_batch(batch_size)

                # Calculate targets
                target_q = self.critic.predict_target(
                    s2_batch, self.actor.predict_target(s2_batch))

                # Y represents our model targets.
                # B represents our critic targets.
                y_coll_i = np.zeros((batch_size, self.horizon))
                b_coll_i = np.zeros(batch_size)
                for k in range(batch_size):
                    y_coll_i[k, :] = r_batch[k, :, 0]
                    if t_batch[k]:
                        b_coll_i[k] = r_batch[k, 0, 0]
                    else:
                        b_coll_i[k] = np.mean(target_q[k, :self.horizon])

                # Update the model and critic given the targets.
                (loss, model_loss, expected_reward) = self.critic.train(
                    s_batch, a_batch,
                    np.reshape(y_coll_i, (batch_size, self.horizon)),
                    np.reshape(b_coll_i, (batch_size, 1)))

                # Update the actor policy using the sampled gradient.
                a_outs = self.actor.predict(s_batch)
                grads = self.critic.action_gradients(s_batch, a_outs)
                self.actor.train(s_batch, grads[0])

                # Update target networks
                self.actor.update_target_network()
                self.critic.update_target_network()

                # Output training statistics.
                if ((optimization_step % 20 == 0) or
                    (optimization_step == optimization_steps - 1)):
                    print("[%d] Loss: %.4f, Exp Reward: %.4f" %
                          (optimization_step, loss, expected_reward))

            # Write episode summary statistics.
            summary_str = self.sess.run(
                summary_ops,
                feed_dict={
                    summary_vars[0]: average_epoch_reward,
                    summary_vars[1]: loss,
                    summary_vars[2]: expected_reward,
                })
            writer.add_summary(summary_str, epoch)
            writer.flush()

            # Save model checkpoint.
            print("Saving model checkpoint: %s/model.ckpt-%d.meta" %
                  (summary_dir, epoch))
            saver.save(
                self.sess, summary_dir + '/model.ckpt', global_step=epoch)
            self.sess.run(increment_global_step)


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

    def get_action_dim(self):
        return self.actions.shape[2]

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork:

    def __init__(self,
                 sess,
                 create_critic_network,
                 horizon,
                 num_actor_vars,
                 collision_weight=0.01,
                 uncertainty_weight=1.0,
                 tau=0.001,
                 learning_rate=0.001):
        self.sess = sess
        self.uncertainty_weight = uncertainty_weight

        # Critic network.
        (self.inputs, self.actions, self.y_coll_out,
         self.b_coll_out) = create_critic_network("critic_source")
        network_params = tf.trainable_variables()[num_actor_vars:]
        print("Critic network has %s parameters." % np.sum(
            [v.get_shape().num_elements() for v in network_params]))

        # Target network.
        (self.target_inputs, self.target_actions, self.target_y_coll_out,
         self.target_b_coll_out) = create_critic_network("critic_target")
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

        # Define loss and optimization Op
        self.model_loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.predicted_y_coll_value, logits=self.y_coll_out),
            axis=1)
        self.reward_loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.predicted_b_coll_value, logits=self.b_coll_out),
            axis=1)
        self.loss = tf.reduce_mean(self.model_loss + self.reward_loss)

        self.optimize = tf.train.AdamOptimizer(learning_rate).minimize(
            self.loss)

        # Metrics
        self.expected_reward = tf.reduce_mean(tf.nn.sigmoid(self.b_coll_out))

        # Get the gradient of the net w.r.t. the action
        # critic_influence = self.b_task_out + collision_weight * self.b_coll_out
        critic_influence = self.b_coll_out
        self.action_grads = tf.gradients(critic_influence, self.actions)

    def train(self, inputs, actions, y_coll, b_coll):
        (loss, model_loss, expected_reward, _) = self.sess.run(
            [self.loss, self.model_loss, self.expected_reward, self.optimize],
            feed_dict={
                self.inputs: inputs,
                self.actions: actions,
                self.predicted_y_coll_value: y_coll,
                self.predicted_b_coll_value: b_coll,
            })
        return (loss, model_loss, expected_reward)

    def predict(self, inputs, actions, bootstraps=50):
        preds = []
        for b in range(bootstraps):
            preds.append(
                self.sess.run(
                    [self.y_coll_out, self.b_coll_out],
                    feed_dict={
                        self.inputs: inputs,
                        self.actions: actions
                    }))
        y_coll_out = np.array([pred[0] for pred in preds])
        b_coll_out = np.array([pred[1] for pred in preds])
        preds = np.concatenate([y_coll_out, b_coll_out], axis=-1)
        preds = 1.0 / (1.0 + np.exp(-preds))
        expectation = np.mean(preds, axis=0)
        stddev = np.std(preds, axis=0)
        preds = (expectation - self.uncertainty_weight * stddev)
        preds = np.clip(preds, 0, 1)
        return preds

    def predict_target(self, inputs, actions, bootstraps=50):
        preds = []
        for b in range(bootstraps):
            preds.append(
                self.sess.run(
                    [self.target_y_coll_out, self.target_b_coll_out],
                    feed_dict={
                        self.target_inputs: inputs,
                        self.target_actions: actions
                    }))
        y_coll_out = np.array([pred[0] for pred in preds])
        b_coll_out = np.array([pred[1] for pred in preds])
        preds = np.concatenate([y_coll_out, b_coll_out], axis=-1)
        preds = 1.0 / (1.0 + np.exp(-preds))
        expectation = np.mean(preds, axis=0)
        stddev = np.std(preds, axis=0)
        preds = (expectation - self.uncertainty_weight * stddev)
        preds = np.clip(preds, 0, 1)
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

    def __init__(self, mu, sigma=0.2, theta=0.15, dt=0.25, x0=None):
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
