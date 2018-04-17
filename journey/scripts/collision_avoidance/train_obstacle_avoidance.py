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
import tf as tf2
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
from model_evaluator import ModelValidator


class DeepDronePlanner:

    def __init__(self,
                 distance_threshold=0.5,
                 rate=4,
                 discrete_controls=True,
                 use_probability=True):
        self.distance_threshold = distance_threshold  # meters
        self.update_rate = rate  # Hz
        self.discrete_controls = discrete_controls
        self.use_probability = use_probability

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
        self.pose = Pose()
        self.nav_goal = Pose()

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

        # Velocity control scaling constant.
        self.forward_kp = 0.6
        self.forward_ki = 0.001
        self.forward_kd = 0.1

        # Gaz PID variables.
        self.up_kp = 0.6
        self.up_ki = 0.1
        self.up_kd = 0.001

        # Yaw PID variables.
        self.yaw_kp = 0.6
        self.yaw_ki = 0.001
        self.yaw_kd = 0.1

        # Set up policy search network.
        self.linear_velocity = 0.5
        if self.discrete_controls:
            self.atoms = 5
            self.action_dim = self.atoms**2
        else:
            self.action_dim = 2
        scale = 0.1
        self.image_width = int(640 * scale)
        self.image_height = int(360 * scale)
        self.sequence_length = 4
        self.horizon = 16
        self.frame_buffer = deque(maxlen=self.sequence_length)
        self.backup_velocity = 0.5
        self.ddpg = DeepDeterministicPolicyGradients(
            self.create_network,
            horizon=self.horizon,
            discrete_controls=self.discrete_controls,
            use_probability=self.use_probability)

        print("Deep drone planner initialized.")

    def on_new_image(self, image):
        self.image_msg = image

    def on_new_contact_data(self, contact):
        # Surprisingly, this works pretty well for collision detection
        if len(contact.states) > 0:
            self.collided = True

    def on_new_state(self, state):
        self.pose = state.pose.pose

    def create_network(self, scope):
        num_params = len(tf.trainable_variables())
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
        actor_params = tf.trainable_variables()[num_params:]

        # Predict actions.
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
        if self.discrete_controls:
            actions = tf.nn.softmax(actions)
        else:
            actions = tf.nn.tanh(actions)

        lstm_outputs = tf.concat([lstm_outputs, actions], axis=-1)

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

        return inputs, actions, y_coll, b_coll, actor_params

    def get_current_frame(self):
        image_data = ros_numpy.numpify(self.image_msg)
        r, g, b = image_data[:, :, 0], image_data[:, :, 1], image_data[:, :, 2]
        greyscale = 0.2989 * r + 0.5870 * g + 0.1140 * b
        frame = scipy.misc.imresize(
            greyscale, [self.image_height, self.image_width], mode='F')
        return (frame - np.std(frame)) / np.mean(frame)

    def get_current_state(self):
        frame = self.get_current_frame()
        self.frame_buffer.append(frame)
        while len(self.frame_buffer) < self.frame_buffer.maxlen:
            self.rate.sleep()
            frame = self.get_current_frame()
            self.frame_buffer.append(frame)
        return np.stack(list(self.frame_buffer), axis=-1)

    def reset(self, start=(0, 0, 0), goal=(0, 0, 0), training=True):
        self.velocity_publisher.publish(Twist())

        # Randomize simulation environment.
        self.randomize_simulation(start=start, training=training)

        # Clear our frame buffer.
        self.frame_buffer.clear()

        # Take-off.
        self.unpause_physics()
        self.takeoff_publisher.publish(EmptyMessage())

        # Get state.
        state = self.get_current_state()

        # Set goal pose.
        if training:
            goal_position = self.randomize_simulation.GetRandomAptPosition()
            self.nav_goal.position.x = goal_position[0]
            self.nav_goal.position.y = goal_position[1]
            self.nav_goal.position.z = goal_position[2]
        else:
            self.nav_goal.position.x = goal[0]
            self.nav_goal.position.y = goal[1]
            self.nav_goal.position.z = goal[2]

        self.forward_integral = 0.0
        self.forward_prior = 0.0
        self.up_integral = 0.0
        self.up_prior = 0.0
        self.yaw_integral = 0.0
        self.yaw_prior = 0.0

        print("Set navigation goal: (%.4f, %.4f, %.4f)" %
              (self.nav_goal.position.x, self.nav_goal.position.y,
               self.nav_goal.position.z))

        # Reset collision state.
        self.collided = False

        return state

    def step(self, state, action):
        control = self.action_to_control(action)
        vel_msg = Twist()

        x = np.array(
            [self.pose.position.x, self.pose.position.y, self.pose.position.z])
        quaternion = (self.pose.orientation.x, self.pose.orientation.y,
                      self.pose.orientation.z, self.pose.orientation.w)
        _, _, yaw = tf2.transformations.euler_from_quaternion(quaternion)
        g = np.array([
            self.nav_goal.position.x, self.nav_goal.position.y,
            self.nav_goal.position.z
        ])

        distance = np.linalg.norm(g[:2] - x[:2])

        # Angular velocity in the XY plane.
        angle = np.arctan2(g[1] - x[1], g[0] - x[0])
        yaw_error = angle - yaw
        self.yaw_integral += yaw_error / self.update_rate
        yaw_derivative = (yaw_error - self.yaw_prior) * self.update_rate
        vel_msg.angular.z = np.clip(
            self.yaw_kp * yaw_error + self.yaw_ki * self.yaw_integral +
            self.yaw_kd * yaw_derivative, -0.5, 0.5)
        self.yaw_prior = yaw_error

        # Linear velocity in the forward axis
        forward_error = distance * np.cos(yaw_error)
        self.forward_integral += forward_error / self.update_rate
        forward_derivative = (
            forward_error - self.forward_prior) * self.update_rate
        vel_msg.linear.x = np.clip(self.forward_kp * forward_error +
                                   self.forward_ki * self.forward_integral +
                                   self.forward_kd * forward_derivative, 0, 1)
        self.forward_prior = forward_error

        # Linear velocity in the up axis.
        up_error = g[2] - x[2]
        self.up_integral += up_error / self.update_rate
        up_derivative = (up_error - self.up_prior) * self.update_rate
        vel_msg.linear.z = np.clip(
            self.up_kp * up_error + self.up_ki * self.up_integral +
            self.up_kd * up_derivative, -0.5, 0.5)
        self.up_prior = up_error

        vel_msg.linear.z += control[0] * 0.5
        vel_msg.angular.z += control[1] * 0.5
        vel_msg.linear.z = np.clip(vel_msg.linear.z, -1, 1)
        vel_msg.angular.z = np.clip(vel_msg.angular.z, -1, 1)
        self.velocity_publisher.publish(vel_msg)

        # Wait.
        self.rate.sleep()

        return self.get_current_state()

    def visualize(self, state, actions):
        # NOTE(kirmani): This doesn't work after adding non-planar controls.
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
        control = self.action_to_control(action)
        collision_reward = 1 if not self.collided else 0
        if self.use_probability:
            return collision_reward
        else:
            return collision_reward

    def terminal(self, state, action):
        x = np.array(
            [self.pose.position.x, self.pose.position.y, self.pose.position.z])
        g = np.array([
            self.nav_goal.position.x, self.nav_goal.position.y,
            self.nav_goal.position.z
        ])

        distance = np.linalg.norm(g - x)
        goal_reached = distance < self.distance_threshold

        terminal = self.collided or goal_reached
        if terminal:
            if goal_reached:
                print("Goal reached!")
            elif self.collided:
                print("Collided :(")
            self.pause_physics()
            self.velocity_publisher.publish(Twist())
        return terminal

    def action_to_control(self, action):
        if self.discrete_controls:
            argmax = np.argmax(action)
            values = 2.0 * np.arange(self.atoms) / (self.atoms - 1.0) - 1.0
            linear_z = values[argmax / self.atoms]
            angular_z = values[argmax % self.atoms]
        else:
            linear_z = action[0]
            angular_z = action[1]
        return (linear_z, angular_z)

    # def control_to_metric(self, control):
    #     # NOTE(kirmani): This doesn't work after adding non-planar controls.
    #     return None

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
            os.path.dirname(__file__), '../../../learning/deep_drone/')
        self.ddpg.train(env, logdir=logdir, model_dir=model_dir)

    def eval(self, model_dir, num_attempts):
        env = Environment(self.reset, self.step, self.reward, self.terminal)
        model_dir = os.path.join(os.getcwd(), model_dir)
        self.ddpg.eval(
            env, model_dir, num_attempts=num_attempts, max_episode_len=1000)

    def test(self, model_dir):
        env = Environment(self.reset, self.step, self.reward, self.terminal)
        model_dir = os.path.join(os.getcwd(), model_dir)
        self.ddpg.load_model(model_dir)

        validator = ModelValidator()
        validator.validate(env, self.ddpg)

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
    elif args.test:
        deep_drone_planner.test(args.model)
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
            '-t',
            '--test',
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
