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
import time
import traceback
from collections import deque
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import ContactsState
from sensor_msgs.msg import Image
from std_msgs.msg import Empty as EmptyMessage
from std_msgs.msg import String
from std_srvs.srv import Empty as EmptyService
from visualization_msgs.msg import Marker
from tum_ardrone.msg import filter_state
from journey.srv import FlyToGoal
from journey.srv import FlyToGoalResponse
from ddpg import DeepDeterministicPolicyGradients
from ddpg import OrnsteinUhlenbeckActionNoise
from ddpg import Environment


class DeepDronePlanner:

    def __init__(self, distance_threshold=0.5, rate=10):
        self.distance_threshold = distance_threshold  # meters
        self.rate = rate  # Hz

        # Initialize our ROS node.
        rospy.init_node('deep_drone_planner', anonymous=True)

        # Inputs.
        self.depth_subscriber = rospy.Subscriber(
            '/ardrone/front/depth/image_raw', Image, self._OnNewDepth)
        self.depth_msg = None


        self.collision_subscriber = rospy.Subscriber(
            '/ardrone/crash_sensor', ContactsState, self._OnNewContactData)
        self.collided = False


        # Actions.
        self.velocity_publisher = rospy.Publisher(
            '/cmd_vel', Twist, queue_size=10)

        # Reset topics.
        self.takeoff_publisher = rospy.Publisher(
            '/ardrone/takeoff', EmptyMessage, queue_size=10)

        # Listen for new goal when planning at test time.
        s = rospy.Service('fly_to_goal', FlyToGoal, self.FlyToGoal)

        # The rate which we publish commands.
        self.rate = rospy.Rate(self.rate)

        # Set up policy search network.
        self.action_dim = 2
        scale = 0.05
        self.image_width = int(640 * scale)
        self.image_height = int(360 * scale)
        self.sequence_length = 4
        self.frame_buffer = deque(maxlen=self.sequence_length)
        self.ddpg = DeepDeterministicPolicyGradients(self.create_actor_network,
                                                     self.create_critic_network)

        print("Deep drone planner initialized.")

    def _OnNewDepth(self, depth):
        self.depth_msg = depth

    def _OnNewContactData(self, contact):
        # Surprisingly, this works pretty well for collision detection
        self.collided = len(contact.states) > 0

    def FlyToGoal(self, req):
        self.goal_pose.position.x = self.pose.position.x + req.x
        self.goal_pose.position.y = self.pose.position.y + req.y
        self.goal_pose.position.z = self.pose.position.z + req.z
        print("Flying to: (%s, %s, %s)" %
              (self.goal_pose.position.x, self.goal_pose.position.y,
               self.goal_pose.position.z))
        return FlyToGoalResponse(True)

    def create_actor_network(self, scope):
        inputs = tf.placeholder(tf.float32, (None, self.image_height, self.image_width, self.sequence_length))
        depth = tf.contrib.layers.conv2d(
            inputs,
            num_outputs=32,
            activation_fn=None,
            kernel_size=(5, 5),
            stride=(2, 2),
            weights_regularizer=tf.nn.l2_loss)
        depth = tf.contrib.layers.batch_norm(depth)
        depth = tf.nn.relu(depth)
        depth = tf.contrib.layers.conv2d(
            depth,
            num_outputs=32,
            activation_fn=None,
            kernel_size=(5, 5),
            stride=(2, 2),
            weights_regularizer=tf.nn.l2_loss)
        depth = tf.contrib.layers.batch_norm(depth)
        depth = tf.nn.relu(depth)
        depth = tf.contrib.layers.conv2d(
            depth,
            num_outputs=32,
            activation_fn=None,
            kernel_size=(5, 5),
            stride=(1, 1),
            weights_regularizer=tf.nn.l2_loss)
        depth = tf.contrib.layers.batch_norm(depth)
        depth = tf.nn.relu(depth)
        depth = tf.contrib.layers.flatten(depth)
        depth = tf.contrib.layers.fully_connected(
            depth, 200, activation_fn=None, weights_regularizer=tf.nn.l2_loss)
        depth = tf.contrib.layers.batch_norm(depth)
        depth = tf.nn.relu(depth)
        depth = tf.contrib.layers.fully_connected(
            depth, 200, activation_fn=None, weights_regularizer=tf.nn.l2_loss)
        depth = tf.contrib.layers.batch_norm(depth)
        depth = tf.nn.relu(depth)
        action_weights = tf.Variable(tf.random_uniform([200, self.action_dim], -3e-4, 3e-4))
        action_bias = tf.Variable(tf.random_uniform([self.action_dim], -3e-4, 3e-4))
        actions = tf.matmul(depth, action_weights) + action_bias
        actions = tf.nn.tanh(actions)
        return inputs, actions

    def create_critic_network(self, scope):
        inputs = tf.placeholder(tf.float32, (None, self.image_height, self.image_width, self.sequence_length))
        actions = tf.placeholder(tf.float32, (None, self.action_dim))
        depth = tf.contrib.layers.conv2d(
            inputs,
            num_outputs=32,
            activation_fn=None,
            kernel_size=(5, 5),
            stride=(2, 2),
            weights_regularizer=tf.nn.l2_loss)
        depth = tf.contrib.layers.batch_norm(depth)
        depth = tf.nn.relu(depth)
        depth = tf.contrib.layers.conv2d(
            depth,
            num_outputs=32,
            activation_fn=None,
            kernel_size=(5, 5),
            stride=(2, 2),
            weights_regularizer=tf.nn.l2_loss)
        depth = tf.contrib.layers.batch_norm(depth)
        depth = tf.nn.relu(depth)
        depth = tf.contrib.layers.conv2d(
            depth,
            num_outputs=32,
            activation_fn=None,
            kernel_size=(5, 5),
            stride=(1, 1),
            weights_regularizer=tf.nn.l2_loss)
        depth = tf.contrib.layers.batch_norm(depth)
        depth = tf.nn.relu(depth)
        depth = tf.contrib.layers.flatten(depth)
        depth = tf.concat([depth, actions], axis=-1)
        depth = tf.contrib.layers.fully_connected(
            depth, 200, activation_fn=None, weights_regularizer=tf.nn.l2_loss)
        depth = tf.contrib.layers.batch_norm(depth)
        depth = tf.nn.relu(depth)
        depth = tf.contrib.layers.fully_connected(
            depth, 200, activation_fn=None, weights_regularizer=tf.nn.l2_loss)
        depth = tf.contrib.layers.batch_norm(depth)
        depth = tf.nn.relu(depth)
        out_weights = tf.Variable(tf.random_uniform([200, 1], -3e-4, 3e-4))
        out_bias = tf.Variable(tf.random_uniform([1], -3e-4, 3e-4))
        out = tf.matmul(depth, out_weights) + out_bias
        return inputs, actions, out

    def get_current_frame(self):
        depth_data = ros_numpy.numpify(self.depth_msg)
        depth_data[np.isnan(depth_data)] = 0.0

        depth = scipy.misc.imresize(
            depth_data, [self.image_height, self.image_width],
            mode='F')
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
        # Stop moving our drone
        vel_msg = Twist()
        vel_msg.linear.x = 0
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.z = 0
        self.velocity_publisher.publish(vel_msg)

        # Reset our simulation.
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            reset_world = rospy.ServiceProxy('/gazebo/reset_world',
                                             EmptyService)
            reset_world()
            rospy.sleep(1.)
        except rospy.ServiceException:
            print("Failed to reset simulator.")

        # Take-off.
        self.takeoff_publisher.publish(EmptyMessage())

        # Clear our frame buffer.
        self.frame_buffer.clear()
        state = self.get_current_state()

        return state

    def step(self, state, action):
        vel_msg = Twist()
        vel_msg.linear.x = action[0]
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.z = action[1]
        self.velocity_publisher.publish(vel_msg)

        # Wait.
        self.rate.sleep()

        next_state = self.get_current_state()

        # DEBUG.
        # for i in range(4):
        #     plt.subplot(2, 4, i + 1)
        #     plt.imshow(state[:, :, i])
        #     plt.title('state_%d' % i)
        #     plt.subplot(2, 4, i + 5)
        #     plt.imshow(next_state[:, :, i])
        #     plt.title('next_state_%d' % i)
        # plt.show()
        # exit()

        return next_state

    def reward(self, state, action):
        linear_velocity = action[0] if action[0] > 0 else -1
        angular_velocity = action[1]
        farthest_obstacle = np.amax(state[:, :, -1])
        farthest_obstacle_weight = 0.1
        threshold = 1.0
        distance_reward = farthest_obstacle_weight * (farthest_obstacle - threshold)
        # print(distance_reward)
        return (linear_velocity * np.cos(angular_velocity * np.pi / 2) + distance_reward
                if not self.collided else - 100)

    def terminal(self, state, action):
        return self.collided

    def RunModel(self, model_name, num_attempts):
        env = Environment(self.reset, self.step, self.reward, self.terminal)
        modeldir = os.path.join(
            os.path.dirname(__file__),
            '../../../learning/deep_drone/' + model_name)
        self.ddpg.RunModel(
            env, modeldir, num_attempts=num_attempts, max_episode_len=1000)

    def Train(self, prev_model):
        env = Environment(self.reset, self.step, self.reward, self.terminal)
        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_dim))
        modeldir = None
        if prev_model != None:
            modeldir = os.path.join(
                os.path.dirname(__file__),
                '../../../learning/deep_drone/' + prev_model)
            print("modeldir is %s" % modeldir)
        logdir = os.path.join(
            os.path.dirname(__file__), '../../../learning/deep_drone/')
        self.ddpg.Train(
            env, logdir=logdir, episodes_in_epoch=1, num_epochs=(16 * 200), actor_noise=actor_noise, epsilon_zero=0, model_dir=modeldir, max_episode_len=1000)


def main(args):
    deep_drone_planner = DeepDronePlanner()
    if args.model:
        attempts = 1
        if args.num_attempts:
            attempts = int(args.num_attempts)
        deep_drone_planner.RunModel(args.model, num_attempts=attempts)
    else:
        deep_drone_planner.Train(args.prev_model)


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
