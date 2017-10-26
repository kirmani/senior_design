#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Sean Kirmani <sean@kirmani.io>
#
# Distributed under terms of the MIT license.
"""
DeepDrone trajectory planner.
"""
import argparse
import numpy as np
import random
import rospy
import ros_numpy
import os
import sys
import tensorflow as tf
import traceback
import time
from collections import deque
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
from std_msgs.msg import Empty
from tum_ardrone.msg import filter_state
from journey.srv import FlyToGoal
from journey.srv import FlyToGoalResponse

RATE = 10  # Hz


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
        self.deque.clear()
        self.count = 0


class ActorNetwork:

    def __init__(self,
                 sess,
                 num_inputs,
                 num_actions,
                 tau=0.001,
                 learning_rate=0.0001):
        self.sess = sess
        self.num_inputs = num_inputs
        self.num_actions = num_actions

        # Actor network.
        self.inputs, self.actions = self.create_actor_network()
        network_params = tf.trainable_variables()

        # Target network.
        self.target_inputs, self.target_actions = self.create_actor_network()
        target_network_params = tf.trainable_variables()[len(network_params):]

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [target_network_params[i].assign(tf.multiply(network_params[i], tau) + \
                tf.multiply(target_network_params[i], 1. - tau))
                for i in range(len(target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, num_actions])

        # Combine the gradients here
        self.actor_gradients = tf.gradients(self.actions, network_params,
                                            -self.action_gradient)

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(learning_rate).\
            apply_gradients(zip(self.actor_gradients, network_params))

        self.num_trainable_vars = len(network_params) + len(
            target_network_params)

    def create_actor_network(self):
        inputs = tf.placeholder(tf.float32, (None, self.num_inputs))
        x = tf.contrib.layers.fully_connected(inputs, 32)
        x = tf.contrib.layers.fully_connected(x, 16)
        actions = tf.contrib.layers.fully_connected(
            x, self.num_actions, activation_fn=tf.tanh)
        return inputs, actions

    def train(self, inputs, a_gradient):
        self.sess.run(
            self.optimize,
            feed_dict={self.inputs: inputs,
                       self.action_gradient: a_gradient})

    def predict(self, inputs):
        return self.sess.run(self.actions, feed_dict={self.inputs: inputs})

    def predict_target(self, inputs):
        return self.sess.run(
            self.target_actions, feed_dict={self.target_inputs: inputs})

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork:

    def __init__(self,
                 sess,
                 num_inputs,
                 num_actions,
                 num_actor_vars,
                 tau=0.001,
                 learning_rate=0.001):
        self.sess = sess
        self.num_inputs = num_inputs
        self.num_actions = num_actions

        # Critic network.
        (self.inputs, self.actions, self.out) = self.create_critic_network()
        network_params = tf.trainable_variables()[num_actor_vars:]

        # Target network.
        (self.target_inputs, self.target_actions,
         self.target_out) = self.create_critic_network()
        target_network_params = tf.trainable_variables()[(
            len(network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [target_network_params[i].assign(tf.multiply(network_params[i], tau) + \
                tf.multiply(target_network_params[i], 1. - tau))
                for i in range(len(target_network_params))]

        # Network target (y_i)
        # Obtained from the target networks
        self.predicted_q_value = tf.placeholder(tf.float32, (None, 1))

        # Define loss and optimization Op
        self.loss = tf.losses.mean_squared_error(self.out,
                                                 self.predicted_q_value)
        self.optimize = tf.train.AdamOptimizer(learning_rate).minimize(
            self.loss)

        # Get the gradient of the net w.r.t. the action
        self.action_grads = tf.gradients(self.out, self.actions)

    def create_critic_network(self):
        inputs = tf.placeholder(tf.float32, (None, self.num_inputs))
        actions = tf.placeholder(tf.float32, (None, self.num_actions))
        x = tf.contrib.layers.fully_connected(inputs, 32)
        x = tf.concat([x, actions], axis=-1)
        x = tf.contrib.layers.fully_connected(x, 16)
        out = tf.contrib.layers.fully_connected(x, 1, activation_fn=None)
        return inputs, actions, out

    def train(self, inputs, actions, reward):
        return self.sess.run(
            [self.out, self.optimize],
            feed_dict={
                self.inputs: inputs,
                self.actions: actions,
                self.predicted_q_value: reward
            })

    def predict(self, inputs, actions):
        return self.sess.run(
            self.out, feed_dict={self.inputs: inputs,
                                 self.actions: actions})

    def predict_target(self, inputs, actions):
        return self.sess.run(
            self.target_out,
            feed_dict={
                self.target_inputs: inputs,
                self.target_actions: actions
            })

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


class DeepDronePlanner:

    def __init__(self, minibatch_size=32, gamma=0.99, max_episode_len=100):

        rospy.init_node('deep_drone_planner', anonymous=True)
        self.velocity_publisher = rospy.Publisher(
            '/cmd_vel', Twist, queue_size=10)

        # TODO(kirmani): Query the depth image instead of the RGB image.
        self.takeoff_publisher = rospy.Publisher(
            '/ardrone/takeoff', Empty, queue_size=10)
        self.image_subscriber = rospy.Subscriber('/ardrone/front/image_raw',
                                                 Image, self._OnNewImage)
        self.pose_subscriber = rospy.Subscriber('/ardrone/predictedPose',
                                                filter_state, self._OnNewPose)

        s = rospy.Service('fly_to_goal', FlyToGoal, self.FlyToGoal)
        self.pose = Pose()
        self.rate = rospy.Rate(RATE)
        self.image_msg = None

        # Initialize goal
        self.goal_pose = Pose()
        self.goal_pose.position.x = 0
        self.goal_pose.position.y = 0
        self.goal_pose.position.z = 1

        # Start tensorflow session.
        self.sess = tf.Session()

        # Create actor network.
        self.num_inputs = 3
        self.num_actions = 4
        self.actor = ActorNetwork(self.sess, self.num_inputs, self.num_actions)
        self.critic = CriticNetwork(self.sess, self.num_inputs,
                                    self.num_actions,
                                    self.actor.get_num_trainable_vars())
        self.replay_buffer = ReplayBuffer()
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.max_episode_len = max_episode_len
        self.actor_noise = OrnsteinUhlenbeckActionNoise(
            mu=np.zeros(self.num_actions))

        self.sess.run(tf.global_variables_initializer())

        # Initialize target network weights
        self.actor.update_target_network()
        self.critic.update_target_network()

        # Initialize policy with heuristic.
        # self._InitializePolicy()

        print("Deep drone planner initialized.")

    def _InitializePolicy(self, num_samples=10000, bounds=5.0, num_epochs=100):
        delta = (np.random.uniform(size=(num_samples, 3)) - 0.5) * (2 * bounds)
        actions = np.zeros((num_samples, 4))
        actions[:, 0:3] = delta / bounds
        rewards = np.linalg.norm(delta, axis=1, keepdims=True)
        for epoch in range(num_epochs):
            self.critic.train(delta, actions, rewards)
            self.actor.train(delta, actions)
        print("Policy initialized.")

    def _OnNewPose(self, data):
        self.pose.position.x = round(data.x, 4)
        self.pose.position.y = round(data.y, 4)
        self.pose.position.z = round(data.z, 4)

    def _OnNewImage(self, image):
        self.image_msg = image

    def FlyToGoal(self, req):
        self.goal_pose.position.x = self.pose.position.x + req.x
        self.goal_pose.position.y = self.pose.position.y + req.y
        self.goal_pose.position.z = self.pose.position.z + req.z
        print("Flying to: (%s, %s, %s)" %
              (self.goal_pose.position.x, self.goal_pose.position.y,
               self.goal_pose.position.z))
        return FlyToGoalResponse(True)

    def build_summaries(self):
        episode_reward = tf.Variable(0.)
        tf.summary.scalar("Reward", episode_reward)
        episode_ave_max_q = tf.Variable(0.)
        tf.summary.scalar("Qmax_Value", episode_ave_max_q)

        summary_vars = [episode_reward, episode_ave_max_q]
        summary_ops = tf.summary.merge_all()

        return summary_ops, summary_vars

    def Train(self, distance_threshold=0.5):
        # Set up summary Ops
        summary_ops, summary_vars = self.build_summaries()

        # Create a new log directory (if you run low on disk space you can either disable this or delete old logs)
        # run: `tensorboard --logdir log` to see all the nice summaries
        for n_model in range(1000):
            dirname = os.path.dirname(__file__)
            summary_dir = os.path.join(
                dirname, '../../../learning/deep_drone/model_%d' % n_model)
            if not os.path.exists(summary_dir):
                break
        writer = tf.summary.FileWriter(summary_dir, self.sess.graph)

        # Initialize velocity message.
        vel_msg = Twist()

        self.takeoff_publisher.publish(Empty())
        start_time = time.time()
        i = 0
        while not rospy.is_shutdown():
            # if not self.image_msg:
            #     print("No image available.")
            # else:
            #     # Create inputs for network.
            #     image = ros_numpy.numpify(self.image_msg)
            # Set new goal.
            bounds = 0.5
            new_goal = (np.random.uniform(size=(3)) - 0.5) * (2 * bounds)
            new_goal[2] += (bounds + 1)
            # print("New goal: %s" % new_goal)
            self.goal_pose.position.x = new_goal[0]
            self.goal_pose.position.y = new_goal[1]
            self.goal_pose.position.z = new_goal[2]
            goal = np.array([
                self.goal_pose.position.x, self.goal_pose.position.y,
                self.goal_pose.position.z
            ])
            x = np.array([
                self.pose.position.x, self.pose.position.y, self.pose.position.z
            ])

            # Set the initial state.
            state = goal - x

            ep_reward = 0.0
            ep_ave_max_q = 0.0

            for j in range(self.max_episode_len):
                # Added exploration noise.
                action = self.actor.predict(
                    np.stack([state]))[0] + self.actor_noise()

                vel_msg.linear.x = action[0]
                vel_msg.linear.y = action[1]
                vel_msg.linear.z = action[2]
                vel_msg.angular.z = action[3]
                self.velocity_publisher.publish(vel_msg)

                # Wait.
                self.rate.sleep()

                # Get next state.
                x = np.array([
                    self.pose.position.x, self.pose.position.y,
                    self.pose.position.z
                ])
                next_state = goal - x

                # Get reward.
                distance = np.linalg.norm(next_state)
                terminal = ((distance < distance_threshold) or
                            (j == self.max_episode_len - 1))
                reward = np.exp(-distance)

                self.replay_buffer.add(state, action, reward, terminal,
                                       next_state)

                if self.replay_buffer.size() > self.minibatch_size:
                    s_batch, a_batch, r_batch, t_batch, s2_batch = \
                        self.replay_buffer.sample_batch(self.minibatch_size)

                    # Calculate targets
                    target_q = self.critic.predict_target(
                        s2_batch, self.actor.predict_target(s2_batch))

                    y_i = []
                    for k in range(self.minibatch_size):
                        if t_batch[k]:
                            y_i.append(r_batch[k])
                        else:
                            y_i.append(r_batch[k] + self.gamma * target_q[k])

                    # Update the critic given the targets
                    predicted_q_value, _ = self.critic.train(
                        s_batch, a_batch,
                        np.reshape(y_i, (self.minibatch_size, 1)))

                    ep_ave_max_q += np.amax(predicted_q_value)

                    # Update the actor policy using the sampled gradient
                    a_outs = self.actor.predict(s_batch)
                    grads = self.critic.action_gradients(s_batch, a_outs)
                    self.actor.train(s_batch, grads[0])

                    # Update target networks
                    self.actor.update_target_network()
                    self.critic.update_target_network()

                state = next_state
                ep_reward += reward

                if terminal:
                    summary_str = self.sess.run(
                        summary_ops,
                        feed_dict={
                            summary_vars[0]: ep_reward,
                            summary_vars[1]: ep_ave_max_q / float(j + 1)
                        })

                    writer.add_summary(summary_str, i)
                    writer.flush()

                    print('| Reward: {:4f} | Episode: {:d} | Qmax: {:.4f}'.format(ep_reward, \
                        i, (ep_ave_max_q / float(j))))

                    i += 1
                    break

            # Wait.
            self.rate.sleep()


def main(args):
    deep_drone_planner = DeepDronePlanner()
    deep_drone_planner.Train()


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
