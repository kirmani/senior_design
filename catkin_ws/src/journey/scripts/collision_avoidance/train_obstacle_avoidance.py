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
import matplotlib.pyplot as plt
import os
import rospy
import ros_numpy
import scipy
import sys
import tensorflow as tf
import tf as transform
import time
import traceback
from collections import deque
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import ContactsState
from gazebo_msgs.msg import ModelState
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import Empty as EmptyMessage
from std_msgs.msg import String
from std_srvs.srv import Empty as EmptyService
from visualization_msgs.msg import Marker
from tum_ardrone.msg import filter_state
from ddpg import DeepDeterministicPolicyGradients
from ddpg import OrnsteinUhlenbeckActionNoise
from ddpg import Environment
from multiplicative_integration_lstm import MultiplicativeIntegrationLSTMCell


class DeepDronePlanner:

    def __init__(self,
                 distance_threshold=0.5,
                 rate=4,
                 episodes_before_position_reset=5):
        self.distance_threshold = distance_threshold  # meters
        self.rate = rate  # Hz

        # Initialize our ROS node.
        rospy.init_node('deep_drone_planner', anonymous=True)

        # Inputs.
        self.depth_subscriber = rospy.Subscriber(
            '/ardrone/front/depth/image_raw', Image, self.on_new_depth)
        self.depth_msg = None

        # Subscribe to ground truth pose.
        self.ground_truth_subscriber = rospy.Subscriber(
            '/ground_truth/state', Odometry, self.on_new_state)
        self.last_collision_pose = Pose()
        self.pose = None

        # Subscribe to collision detector.
        self.collision_subscriber = rospy.Subscriber(
            '/ardrone/crash_sensor', ContactsState, self.on_new_contact_data)
        self.collided = False

        # Actions.
        self.velocity_publisher = rospy.Publisher(
            '/cmd_vel', Twist, queue_size=10)

        # Reset topics.
        self.takeoff_publisher = rospy.Publisher(
            '/ardrone/takeoff', EmptyMessage, queue_size=10)

        self.land_publisher = rospy.Publisher(
            '/ardrone/land', EmptyMessage, queue_size=10)

        # Publish model state.
        self.model_state_publisher = rospy.Publisher('/gazebo/set_model_state',
                                                     ModelState)

        # The rate which we publish commands.
        self.rate = rospy.Rate(self.rate)

        # Set up policy search network.
        self.action_dim = 1
        scale = 0.1
        self.image_width = int(640 * scale)
        self.image_height = int(360 * scale)
        self.sequence_length = 4
        self.horizon = 16
        self.frame_buffer = deque(maxlen=self.sequence_length)
        self.linear_velocity = 0.5
        self.ddpg = DeepDeterministicPolicyGradients(
            self.create_actor_network,
            self.create_critic_network,
            self.action_dim,
            horizon=self.horizon)

        print("Deep drone planner initialized.")

    def on_new_depth(self, depth):
        self.depth_msg = depth

    def on_new_contact_data(self, contact):
        # Surprisingly, this works pretty well for collision detection
        if len(contact.states) > 0:
            self.collided = True

    def on_new_state(self, state):
        self.pose = state.pose.pose

    def create_actor_network(self, scope):
        inputs = tf.placeholder(tf.float32,
                                (None, self.image_height, self.image_width,
                                 self.sequence_length))
        depth = tf.contrib.layers.conv2d(
            inputs,
            num_outputs=32,
            activation_fn=None,
            kernel_size=(8, 8),
            stride=(4, 4),
            weights_regularizer=tf.nn.l2_loss)
        depth = tf.contrib.layers.batch_norm(depth)
        depth = tf.nn.relu(depth)
        depth = tf.contrib.layers.conv2d(
            depth,
            num_outputs=32,
            activation_fn=None,
            kernel_size=(4, 4),
            stride=(2, 2),
            weights_regularizer=tf.nn.l2_loss)
        depth = tf.contrib.layers.batch_norm(depth)
        depth = tf.nn.relu(depth)
        depth = tf.contrib.layers.conv2d(
            depth,
            num_outputs=32,
            activation_fn=None,
            kernel_size=(3, 3),
            stride=(1, 1),
            weights_regularizer=tf.nn.l2_loss)
        depth = tf.contrib.layers.batch_norm(depth)
        depth = tf.nn.relu(depth)
        depth = tf.contrib.layers.flatten(depth)
        depth = tf.contrib.layers.fully_connected(
            depth, 256, activation_fn=None, weights_regularizer=tf.nn.l2_loss)
        depth = tf.contrib.layers.batch_norm(depth)
        depth = tf.nn.relu(depth)
        depth = tf.stack([depth for _ in range(self.horizon)], axis=1)
        lstm_inputs = depth
        lstm_cell = MultiplicativeIntegrationLSTMCell(num_units=16)
        lstm_outputs, lstm_states = tf.nn.dynamic_rnn(
            lstm_cell, lstm_inputs, dtype=tf.float32, scope=scope)

        actions = tf.contrib.layers.fully_connected(
            lstm_outputs,
            16,
            activation_fn=None,
            weights_regularizer=tf.nn.l2_loss)
        actions = tf.contrib.layers.batch_norm(actions)
        actions = tf.nn.relu(actions)
        actions = tf.contrib.layers.fully_connected(
            actions,
            self.action_dim,
            activation_fn=None,
            weights_regularizer=tf.nn.l2_loss)
        # action_weights = tf.Variable(
        #     tf.random_uniform([256, self.action_dim], -3e-4, 3e-4))
        # action_bias = tf.Variable(
        #     tf.random_uniform([self.action_dim], -3e-4, 3e-4))
        # actions = tf.matmul(depth, action_weights) + action_bias
        actions = tf.nn.tanh(actions)
        return inputs, actions

    def create_critic_network(self, scope):
        inputs = tf.placeholder(tf.float32,
                                (None, self.image_height, self.image_width,
                                 self.sequence_length))
        actions = tf.placeholder(tf.float32, (None, self.horizon,
                                              self.action_dim))
        depth = tf.contrib.layers.conv2d(
            inputs,
            num_outputs=32,
            activation_fn=None,
            kernel_size=(8, 8),
            stride=(4, 4),
            weights_regularizer=tf.nn.l2_loss)
        depth = tf.contrib.layers.batch_norm(depth)
        depth = tf.nn.relu(depth)
        depth = tf.contrib.layers.conv2d(
            depth,
            num_outputs=32,
            activation_fn=None,
            kernel_size=(4, 4),
            stride=(2, 2),
            weights_regularizer=tf.nn.l2_loss)
        depth = tf.contrib.layers.batch_norm(depth)
        depth = tf.nn.relu(depth)
        depth = tf.contrib.layers.conv2d(
            depth,
            num_outputs=32,
            activation_fn=None,
            kernel_size=(3, 3),
            stride=(1, 1),
            weights_regularizer=tf.nn.l2_loss)
        depth = tf.contrib.layers.batch_norm(depth)
        depth = tf.nn.relu(depth)
        depth = tf.contrib.layers.flatten(depth)
        depth = tf.contrib.layers.fully_connected(
            depth, 256, activation_fn=None, weights_regularizer=tf.nn.l2_loss)
        depth = tf.contrib.layers.batch_norm(depth)
        depth = tf.nn.relu(depth)

        act = tf.contrib.layers.fully_connected(
            actions, 16, activation_fn=None, weights_regularizer=tf.nn.l2_loss)
        act = tf.contrib.layers.batch_norm(act)
        act = tf.nn.relu(act)
        act = tf.contrib.layers.fully_connected(
            act, 16, activation_fn=None, weights_regularizer=tf.nn.l2_loss)
        act = tf.contrib.layers.batch_norm(act)
        act = tf.nn.relu(act)

        depth = tf.stack([depth for _ in range(self.horizon)], axis=1)
        lstm_inputs = tf.concat([depth, act], axis=-1)
        lstm_cell = MultiplicativeIntegrationLSTMCell(num_units=16)
        lstm_outputs, lstm_states = tf.nn.dynamic_rnn(
            lstm_cell, lstm_inputs, dtype=tf.float32, scope=scope)

        y = tf.contrib.layers.fully_connected(
            lstm_outputs,
            16,
            activation_fn=None,
            weights_regularizer=tf.nn.l2_loss)
        y = tf.contrib.layers.batch_norm(y)
        y = tf.nn.relu(y)
        y = tf.contrib.layers.fully_connected(
            y, 1, activation_fn=None, weights_regularizer=tf.nn.l2_loss)
        y = tf.squeeze(y)
        # y_out_weights = tf.Variable(
        #     tf.random_uniform([16, 1], -3e-4, 3e-4))
        # y_out_bias = tf.Variable(tf.random_uniform([16, 1], -3e-4, 3e-4))
        # y = tf.tensordot(y, y_out_weights, [[2], [1]]) # + y_out_bias
        # print(y)
        # exit()

        b = tf.contrib.layers.fully_connected(
            lstm_outputs,
            16,
            activation_fn=None,
            weights_regularizer=tf.nn.l2_loss)
        b = tf.contrib.layers.batch_norm(b)
        b = tf.nn.relu(b)
        b = tf.reshape(b, [-1, 16 * self.horizon])
        b = tf.contrib.layers.fully_connected(
            b, 1, activation_fn=None, weights_regularizer=tf.nn.l2_loss)
        # b_out_weights = tf.Variable(tf.random_uniform([16, self.horizon], -3e-4, 3e-4))
        # b_out_bias = tf.Variable(tf.random_uniform([self.horizon], -3e-4, 3e-4))
        # b = b * b_out_weights + b_out_bias
        return inputs, actions, y, b

    def get_current_frame(self):
        depth_data = ros_numpy.numpify(self.depth_msg)
        depth_data[np.isnan(depth_data)] = 0.0
        # print(depth_data.shape)
        # r, g, b = depth_data[:, :, 0], depth_data[:, :, 1], depth_data[:, :, 2]
        # depth_data = 0.2989 * r + 0.5870 * g + 0.1140 * b
        # plt.imshow(depth_data, cmap="gray")
        # plt.show()
        # exit()

        depth = scipy.misc.imresize(
            depth_data, [self.image_height, self.image_width], mode='F')
        frame = depth
        return frame

    def get_current_state(self):
        frame = self.get_current_frame()
        self.frame_buffer.append(frame)
        while len(self.frame_buffer) < self.frame_buffer.maxlen:
            self.rate.sleep()
            frame = self.get_current_frame()
            self.frame_buffer.append(frame)
        depth = np.stack(list(self.frame_buffer), axis=-1)
        state = depth
        return state

    def reset(self):
        self.velocity_publisher.publish(Twist())

        # Reset our drone.
        model_state = ModelState()
        model_state.model_name = 'quadrotor'
        model_state.reference_frame = 'world'

        position = (self.last_collision_pose.position.x,
                    self.last_collision_pose.position.y,
                    self.last_collision_pose.position.z)

        quaternion = (self.last_collision_pose.orientation.x,
                      self.last_collision_pose.orientation.y,
                      self.last_collision_pose.orientation.z,
                      self.last_collision_pose.orientation.w)
        _, _, yaw = transform.transformations.euler_from_quaternion(quaternion)
        position = (position[0], position[1], 1)
        quaternion = transform.transformations.quaternion_from_euler(0, 0, yaw)

        reset_pose = Pose()
        reset_pose.position.x = position[0]
        reset_pose.position.y = position[1]
        reset_pose.position.z = position[2]
        reset_pose.orientation.w = quaternion[3]
        reset_pose.orientation.x = quaternion[0]
        reset_pose.orientation.y = quaternion[1]
        reset_pose.orientation.z = quaternion[2]

        model_state.pose = reset_pose
        self.model_state_publisher.publish(model_state)
        rospy.sleep(1.)

        # Take-off.
        self.takeoff_publisher.publish(EmptyMessage())

        # Clear our frame buffer.
        self.frame_buffer.clear()
        state = self.get_current_state()

        # Reset collision state.
        self.collided = False

        return state

    def step(self, state, action):
        vel_msg = Twist()
        vel_msg.linear.x = self.linear_velocity
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.z = action[0]
        self.velocity_publisher.publish(vel_msg)

        # Wait.
        self.rate.sleep()

        next_state = self.get_current_state()

        # DEBUG.
        # for i in range(4):
        #     plt.subplot(2, 4, i + 1)
        #     plt.imshow(state[:, :, i], cmap="gray")
        #     plt.title('state_%d' % i)
        #     plt.subplot(2, 4, i + 5)
        #     plt.imshow(next_state[:, :, i], cmap="gray")
        #     plt.title('next_state_%d' % i)
        # plt.show()
        # exit()

        return next_state

    def reward(self, state, action):
        return 1 if not self.collided else 0

    def terminal(self, state, action):
        if self.collided:
            vel_msg = Twist()
            vel_msg.linear.x = -self.linear_velocity
            vel_msg.linear.y = 0
            vel_msg.linear.z = 0
            vel_msg.angular.z = 0
            self.velocity_publisher.publish(vel_msg)
            rospy.sleep(2.0)
            self.last_collision_pose = self.pose
            self.velocity_publisher.publish(Twist())
        return self.collided

    def run_model(self, model_name, num_attempts):
        env = Environment(self.reset, self.step, self.reward, self.terminal)
        modeldir = os.path.join(
            os.path.dirname(__file__),
            '../../../../learning/deep_drone/' + model_name)
        self.ddpg.run_model(
            env, modeldir, num_attempts=num_attempts, max_episode_len=1000)

    def train(self, prev_model):
        env = Environment(self.reset, self.step, self.reward, self.terminal)
        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_dim))
        modeldir = None
        if prev_model != None:
            modeldir = os.path.join(
                os.path.dirname(__file__),
                '../../../../learning/deep_drone/' + prev_model)
            print("modeldir is %s" % modeldir)
        logdir = os.path.join(
            os.path.dirname(__file__), '../../../../learning/deep_drone/')
        self.ddpg.train(
            env,
            logdir=logdir,
            episodes_in_epoch=1,
            num_epochs=(16 * 200),
            actor_noise=actor_noise,
            model_dir=modeldir,
            max_episode_len=1000)


def main(args):
    deep_drone_planner = DeepDronePlanner()
    if args.model:
        attempts = 1
        if args.num_attempts:
            attempts = int(args.num_attempts)
        deep_drone_planner.run_model(args.model, num_attempts=attempts)
    else:
        deep_drone_planner.train(args.prev_model)


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
        parser.add_argument(
            '-m', '--model', action='store', help='run specific model')
        parser.add_argument(
            '-n',
            '--num_attempts',
            action='store',
            help='number of attempts to run model for')
        parser.add_argument(
            '-t',
            '--prev_model',
            action='store',
            help='name of existing model to start training with')
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
