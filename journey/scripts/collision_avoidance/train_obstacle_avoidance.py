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
from journey.msg import CollisionState
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import ContactsState
from gazebo_msgs.msg import ModelState
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from simulation_randomization import SimulationRandomizer
from std_msgs.msg import Empty as EmptyMessage
from std_msgs.msg import String
from std_srvs.srv import Empty as EmptyService
from visualization_msgs.msg import Marker
from ddpg import DeepDeterministicPolicyGradients
from environment import Environment
from multiplicative_integration_lstm import MultiplicativeIntegrationLSTMCell


class DeepDronePlanner:

    def __init__(self,
                 distance_threshold=0.5,
                 rate=4,
                 episodes_before_position_reset=5):
        self.distance_threshold = distance_threshold  # meters
        self.update_rate = rate  # Hz

        # Set max linear velocity to 0.5 meters/sec.
        self.max_linear_velocity = 0.5
        rospy.set_param('control_vz_max', self.max_linear_velocity * 1000)
        print("Max linear velocity (mm/s): %s" %
              rospy.get_param('control_vz_max'))

        # Set max angular velocity to 30 degrees/sec.
        self.max_angular_velocity = np.pi / 6.0
        rospy.set_param('euler_angle_max', self.max_angular_velocity)
        print("Max angular velocity (mm/s): %s" %
              rospy.get_param('euler_angle_max'))

        # Initialize our ROS node.
        rospy.init_node('deep_drone_planner', anonymous=True)

        # Inputs.
        self.image_subscriber = rospy.Subscriber('/ardrone/front/image_raw',
                                                 Image, self.on_new_image)
        self.image_msg = None

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

        # Publish the collision state.
        self.collision_state_publisher = rospy.Publisher(
            '/ardrone/collision_state', CollisionState, queue_size=10)

        # The rate which we publish commands.
        self.rate = rospy.Rate(self.update_rate)

        # Simulation reset randomization.
        self.randomize_simulation = SimulationRandomizer()

        # Set up policy search network.
        self.action_dim = 2
        scale = 0.1
        self.image_width = int(640 * scale)
        self.image_height = int(360 * scale)
        self.sequence_length = 4
        self.horizon = 16
        self.frame_buffer = deque(maxlen=self.sequence_length)
        self.backup_velocity = 0.5
        self.ddpg = DeepDeterministicPolicyGradients(
            self.create_actor_network,
            self.create_critic_network,
            horizon=self.horizon)

        print("Deep drone planner initialized.")

    def on_new_image(self, image):
        self.image_msg = image

    def on_new_contact_data(self, contact):
        # Surprisingly, this works pretty well for collision detection
        if len(contact.states) > 0:
            self.collided = True

    def on_new_state(self, state):
        self.pose = state.pose.pose

    def create_actor_network(self, scope):
        inputs = tf.placeholder(
            tf.float32,
            (None, self.image_height, self.image_width, self.sequence_length))
        image = tf.contrib.layers.conv2d(
            inputs,
            num_outputs=32,
            activation_fn=None,
            kernel_size=(8, 8),
            stride=(4, 4),
            weights_regularizer=tf.nn.l2_loss)
        image = tf.contrib.layers.batch_norm(image)
        image = tf.nn.relu(image)
        image = tf.contrib.layers.conv2d(
            image,
            num_outputs=32,
            activation_fn=None,
            kernel_size=(4, 4),
            stride=(2, 2),
            weights_regularizer=tf.nn.l2_loss)
        image = tf.contrib.layers.batch_norm(image)
        image = tf.nn.relu(image)
        image = tf.contrib.layers.conv2d(
            image,
            num_outputs=32,
            activation_fn=None,
            kernel_size=(3, 3),
            stride=(1, 1),
            weights_regularizer=tf.nn.l2_loss)
        image = tf.contrib.layers.batch_norm(image)
        image = tf.nn.relu(image)
        image = tf.contrib.layers.flatten(image)
        image = tf.contrib.layers.fully_connected(
            image, 256, activation_fn=None, weights_regularizer=tf.nn.l2_loss)
        image = tf.contrib.layers.batch_norm(image)
        image = tf.nn.relu(image)
        image = tf.stack([image for _ in range(self.horizon)], axis=1)
        lstm_inputs = image
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
        actions = tf.reshape(actions, [-1, 16 * self.horizon])
        actions_weights = tf.Variable(
            tf.random_uniform(
                [16 * self.horizon, self.horizon * self.action_dim], -3e-4,
                3e-4))
        actions_bias = tf.Variable(
            tf.random_uniform([self.horizon * self.action_dim], -3e-4, 3e-4))
        actions = tf.matmul(actions, actions_weights) + actions_bias
        actions = tf.reshape(actions, [-1, self.horizon, self.action_dim])
        actions = tf.nn.tanh(actions)
        return inputs, actions

    def create_critic_network(self, scope):
        inputs = tf.placeholder(
            tf.float32,
            (None, self.image_height, self.image_width, self.sequence_length))
        actions = tf.placeholder(tf.float32,
                                 (None, self.horizon, self.action_dim))
        image = tf.contrib.layers.conv2d(
            inputs,
            num_outputs=32,
            activation_fn=None,
            kernel_size=(8, 8),
            stride=(4, 4),
            weights_regularizer=tf.nn.l2_loss)
        image = tf.contrib.layers.batch_norm(image)
        image = tf.nn.relu(image)
        image = tf.contrib.layers.conv2d(
            image,
            num_outputs=32,
            activation_fn=None,
            kernel_size=(4, 4),
            stride=(2, 2),
            weights_regularizer=tf.nn.l2_loss)
        image = tf.contrib.layers.batch_norm(image)
        image = tf.nn.relu(image)
        image = tf.contrib.layers.conv2d(
            image,
            num_outputs=32,
            activation_fn=None,
            kernel_size=(3, 3),
            stride=(1, 1),
            weights_regularizer=tf.nn.l2_loss)
        image = tf.contrib.layers.batch_norm(image)
        image = tf.nn.relu(image)
        image = tf.contrib.layers.flatten(image)
        image = tf.contrib.layers.fully_connected(
            image, 256, activation_fn=None, weights_regularizer=tf.nn.l2_loss)
        image = tf.contrib.layers.batch_norm(image)
        image = tf.nn.relu(image)

        act = tf.contrib.layers.fully_connected(
            actions, 16, activation_fn=None, weights_regularizer=tf.nn.l2_loss)
        act = tf.contrib.layers.batch_norm(act)
        act = tf.nn.relu(act)
        act = tf.contrib.layers.fully_connected(
            act, 16, activation_fn=None, weights_regularizer=tf.nn.l2_loss)
        act = tf.contrib.layers.batch_norm(act)
        act = tf.nn.relu(act)

        image = tf.stack([image for _ in range(self.horizon)], axis=1)
        lstm_inputs = tf.concat([image, act], axis=-1)
        lstm_cell = MultiplicativeIntegrationLSTMCell(num_units=16)
        lstm_outputs, lstm_states = tf.nn.dynamic_rnn(
            lstm_cell, lstm_inputs, dtype=tf.float32, scope=scope)

        # Collision probability prediction.
        y_coll = tf.contrib.layers.fully_connected(
            lstm_outputs,
            16,
            activation_fn=None,
            weights_regularizer=tf.nn.l2_loss)
        y_coll = tf.contrib.layers.batch_norm(y_coll)
        y_coll = tf.nn.relu(y_coll)
        y_coll = tf.reshape(y_coll, [-1, 16 * self.horizon])
        y_coll = tf.nn.dropout(y_coll, 0.2)
        y_coll_weights = tf.Variable(
            tf.random_uniform([16 * self.horizon, self.horizon], -3e-4, 3e-4))
        y_coll_bias = tf.Variable(
            tf.random_uniform([self.horizon], -3e-4, 3e-4))
        y_coll = tf.matmul(y_coll, y_coll_weights) + y_coll_bias

        b_coll = tf.contrib.layers.fully_connected(
            lstm_outputs,
            16,
            activation_fn=None,
            weights_regularizer=tf.nn.l2_loss)
        b_coll = tf.contrib.layers.batch_norm(b_coll)
        b_coll = tf.nn.relu(b_coll)
        b_coll = tf.reshape(b_coll, [-1, 16 * self.horizon])
        b_coll_weights = tf.Variable(
            tf.random_uniform([16 * self.horizon, 1], -3e-4, 3e-4))
        b_coll_bias = tf.Variable(tf.random_uniform([1], -3e-4, 3e-4))
        b_coll = tf.matmul(b_coll, b_coll_weights) + b_coll_bias

        # Task reward prediction.
        y_task = tf.contrib.layers.fully_connected(
            lstm_outputs,
            16,
            activation_fn=None,
            weights_regularizer=tf.nn.l2_loss)
        y_task = tf.contrib.layers.batch_norm(y_task)
        y_task = tf.nn.relu(y_task)
        y_task = tf.reshape(y_task, [-1, 16 * self.horizon])
        y_task_weights = tf.Variable(
            tf.random_uniform([16 * self.horizon, self.horizon], -3e-4, 3e-4))
        y_task_bias = tf.Variable(
            tf.random_uniform([self.horizon], -3e-4, 3e-4))
        y_task = tf.matmul(y_task, y_task_weights) + y_task_bias

        b_task = tf.contrib.layers.fully_connected(
            lstm_outputs,
            16,
            activation_fn=None,
            weights_regularizer=tf.nn.l2_loss)
        b_task = tf.contrib.layers.batch_norm(b_task)
        b_task = tf.nn.relu(b_task)
        b_task = tf.reshape(b_task, [-1, 16 * self.horizon])
        b_task_weights = tf.Variable(
            tf.random_uniform([16 * self.horizon, 1], -3e-4, 3e-4))
        b_task_bias = tf.Variable(tf.random_uniform([1], -3e-4, 3e-4))
        b_task = tf.matmul(b_task, b_task_weights) + b_task_bias

        return inputs, actions, y_coll, b_coll, y_task, b_task

    def get_current_frame(self):
        image_data = ros_numpy.numpify(self.image_msg)
        r, g, b = image_data[:, :, 0], image_data[:, :, 1], image_data[:, :, 2]
        greyscale = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return scipy.misc.imresize(
            greyscale, [self.image_height, self.image_width], mode='F')

    def get_current_state(self):
        frame = self.get_current_frame()
        self.frame_buffer.append(frame)
        while len(self.frame_buffer) < self.frame_buffer.maxlen:
            self.rate.sleep()
            frame = self.get_current_frame()
            self.frame_buffer.append(frame)
        return np.stack(list(self.frame_buffer), axis=-1)

    def reset(self):
        self.velocity_publisher.publish(Twist())

        # Randomize simulation environment.
        self.randomize_simulation()

        # Clear our frame buffer.
        self.frame_buffer.clear()

        # Take-off.
        self.unpause_physics()
        self.takeoff_publisher.publish(EmptyMessage())

        # Get state.
        state = self.get_current_state()

        # Reset collision state.
        self.collided = False

        return state

    def step(self, state, action):
        control = self.action_to_control(action)

        vel_msg = Twist()
        vel_msg.linear.x = control[0]
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.z = control[1]
        self.velocity_publisher.publish(vel_msg)

        # Wait.
        self.rate.sleep()

        return self.get_current_state()

    def visualize(self, state, actions):
        for i in range(state.shape[2]):
            plt.subplot(1, state.shape[2] + 1, state.shape[2] - i)
            plt.imshow(state[:, :, i], cmap="gray")
            if i > 0:
                plt.title('Frame at t - %d' % i)
            else:
                plt.title('Frame at t')

        x = np.zeros(actions.shape[0] + 1)
        y = np.zeros(actions.shape[0] + 1)
        forward = np.zeros(actions.shape[0] + 1)
        for t in range(actions.shape[0]):
            (linear, angular) = self.control_to_metric(
                self.action_to_control(actions[t]))
            forward[t + 1] = forward[t] + angular
            x[t + 1] = x[t] + np.cos(forward[t + 1]) * linear
            y[t + 1] = y[t] + np.sin(forward[t + 1]) * linear

        plt.subplot(1, state.shape[2] + 1, state.shape[2] + 1)
        plt.plot(y, x, 'k-', lw=2)
        plt.show()
        exit()

    def reward(self, state, action):
        metric = self.control_to_metric(self.action_to_control(action))
        collision_reward = 1 if not self.collided else 0
        task_reward = metric[0] * np.cos(metric[1])
        return (collision_reward, task_reward)

    def terminal(self, state, action):
        if self.collided:
            self.pause_physics()
            self.last_collision_pose = self.pose
            self.velocity_publisher.publish(Twist())
        return self.collided

    def action_to_control(self, action):
        control = np.zeros(2)
        control[0] = (action[0] + 1.0) / 2.0
        control[1] = action[1]
        return control

    def control_to_metric(self, control):
        metric = np.zeros(2)
        metric[0] = control[0] * self.max_linear_velocity
        metric[1] = control[1] * self.max_angular_velocity
        return metric

    def train(self, model_dir=None):
        env = Environment(
            self.reset,
            self.step,
            self.reward,
            self.terminal,
            visualize=self.visualize)
        if model_dir != None:
            model_dir = os.path.join(os.getcwd(), model_dir)
            print("model_dir is %s" % model_dir)
        logdir = os.path.join(
            os.path.dirname(__file__), '../../../../learning/deep_drone/')
        self.ddpg.train(env, logdir=logdir, model_dir=model_dir)

    def eval(self, model_dir, num_attempts):
        env = Environment(self.reset, self.step, self.reward, self.terminal)
        model_dir = os.path.join(os.getcwd(), model_dir)
        self.ddpg.eval(
            env, model_dir, num_attempts=num_attempts, max_episode_len=1000)

    def plan(self, model_dir):
        # Load our model.
        model_dir = os.path.join(os.getcwd(), model_dir)
        self.ddpg.load_model(model_dir)

        # Take-off.
        self.takeoff_publisher.publish(EmptyMessage())

        # Clear our frame buffer.
        self.frame_buffer.clear()
        state = self.get_current_state()

        while not rospy.is_shutdown():
            # Predict the optimal actions over the horizon and the model and
            # critic metrics over the horizon.
            action_sequence = self.ddpg.actor.predict(
                np.expand_dims(state, axis=0))
            critique = self.ddpg.critic.predict(
                np.expand_dims(state, axis=0), action_sequence)

            # Create and publish collision state message.
            # TODO(kirmani): Determine what other useful information we'd
            # like to include in our collision state estimator.
            collision_state = CollisionState()
            collision_state.horizon = self.horizon
            collision_state.action_dimensionality = self.action_dim
            collision_state.collision_probability = 1.0 - np.mean(
                critique[0, :self.horizon, 0])
            collision_state.action = list(
                self.action_to_control(action_sequence[0, 0, :]))
            self.collision_state_publisher.publish(collision_state)

            # Wait.
            self.rate.sleep()

            state = self.get_current_state()

    def pause_physics(self):
        # Pause physics.
        rospy.wait_for_service('gazebo/pause_physics')
        pause_physics = rospy.ServiceProxy('gazebo/pause_physics', EmptyService)
        pause_physics()

    def unpause_physics(self):
        # Unpause physics.
        rospy.wait_for_service('gazebo/unpause_physics')
        unpause_physics = rospy.ServiceProxy('gazebo/unpause_physics',
                                             EmptyService)
        unpause_physics()


def main(args):
    deep_drone_planner = DeepDronePlanner()
    if args.plan:
        deep_drone_planner.plan(args.model)
    elif args.eval:
        attempts = 1
        if args.num_attempts:
            attempts = int(args.num_attempts)
        deep_drone_planner.eval(args.model, num_attempts=attempts)
    else:
        deep_drone_planner.train(args.model)


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
            '-e',
            '--eval',
            action='store_true',
            default=False,
            help='evaluate model')
        parser.add_argument(
            '-p',
            '--plan',
            action='store_true',
            default=False,
            help='load model and use for publishing and planning')
        parser.add_argument(
            '-n',
            '--num_attempts',
            action='store',
            help='number of attempts to run model for')
        args = parser.parse_args(rospy.myargv()[1:])
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
