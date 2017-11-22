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

        # Initialize visualization markers.
        self.drone_marker = Marker()
        self.drone_marker.header.frame_id = "map"
        self.drone_marker.header.stamp = rospy.get_rostime()
        self.drone_marker.ns = "drone"
        self.drone_marker.id = 1
        self.drone_marker.type = Marker.SPHERE
        self.drone_marker.action = Marker.ADD
        self.drone_marker.pose.position.x = 0
        self.drone_marker.pose.position.y = 0
        self.drone_marker.pose.position.z = 0
        self.drone_marker.pose.orientation.x = 0
        self.drone_marker.pose.orientation.y = 0
        self.drone_marker.pose.orientation.z = 0
        self.drone_marker.pose.orientation.w = 0
        self.drone_marker.scale.x = 0.5
        self.drone_marker.scale.y = 0.5
        self.drone_marker.scale.z = 0.5
        self.drone_marker.color.r = 0
        self.drone_marker.color.g = 1
        self.drone_marker.color.b = 0
        self.drone_marker.color.a = 1
        self.drone_marker.lifetime = rospy.Duration(0)

        self.goal_marker = Marker()
        self.goal_marker.header.frame_id = "map"
        self.goal_marker.header.stamp = rospy.get_rostime()
        self.goal_marker.ns = "goal"
        self.goal_marker.id = 0
        self.goal_marker.type = Marker.SPHERE
        self.goal_marker.action = Marker.ADD
        self.goal_marker.pose.position.x = 2
        self.goal_marker.pose.position.y = 0
        self.goal_marker.pose.position.z = 0
        self.goal_marker.pose.orientation.x = 0
        self.goal_marker.pose.orientation.y = 0
        self.goal_marker.pose.orientation.z = 0
        self.goal_marker.pose.orientation.w = 0
        self.goal_marker.scale.x = self.distance_threshold
        self.goal_marker.scale.y = self.distance_threshold
        self.goal_marker.scale.z = self.distance_threshold
        self.goal_marker.color.r = 1
        self.goal_marker.color.g = 0
        self.goal_marker.color.b = 0
        self.goal_marker.color.a = 1
        self.goal_marker.lifetime = rospy.Duration(0)

        # Inputs.
        self.depth_subscriber = rospy.Subscriber(
            '/ardrone/front/depth/image_raw', Image, self._OnNewDepth)

        self.pose_subscriber = rospy.Subscriber('/ardrone/predictedPose',
                                                filter_state, self._OnNewPose)
        self.pose = Pose()

        self.collision_subscriber = rospy.Subscriber(
            '/ardrone/crash_sensor', ContactsState, self._OnNewContactData)
        self.collided = False

        self.depth_msg = None

        # Actions.
        self.velocity_publisher = rospy.Publisher(
            '/cmd_vel', Twist, queue_size=10)

        # Reset topics.
        self.takeoff_publisher = rospy.Publisher(
            '/ardrone/takeoff', EmptyMessage, queue_size=10)
        self.com_publisher = rospy.Publisher(
            '/tum_ardrone/com', String, queue_size=10)

        # Visualization topics.
        self.marker_publisher = rospy.Publisher(
            'visualization_marker', Marker, queue_size=10)
        # Listen for new goal when planning at test time.
        s = rospy.Service('fly_to_goal', FlyToGoal, self.FlyToGoal)

        # The rate which we publish commands.
        self.rate = rospy.Rate(self.rate)

        # Initialize goal.
        self.goal_pose = Pose()
        self.freshPose = False

        # Initialize visualization.
        self.marker_publisher.publish(self.drone_marker)
        self.marker_publisher.publish(self.goal_marker)

        # Set up policy search network.
        self.state_dim = 3
        self.action_dim = 3
        self.goal_dim = 3
        self.image_width = 84
        self.image_height = 84
        self.sequence_length = 4
        self.frame_buffer = deque(maxlen=self.sequence_length)
        self.num_inputs = (
            (self.image_width * self.image_height + self.state_dim) *
            self.sequence_length + self.goal_dim)
        self.ddpg = DeepDeterministicPolicyGradients(self.create_actor_network,
                                                     self.create_critic_network)

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
        self.freshPose = True
        self.pose.position.x = round(data.x, 4)
        self.pose.position.y = round(data.y, 4)
        self.pose.position.z = round(data.z, 4)
        self.drone_marker.pose.position.x = self.pose.position.x
        self.drone_marker.pose.position.y = self.pose.position.y
        self.drone_marker.pose.position.z = self.pose.position.z
        self.drone_marker.action = Marker.MODIFY
        self.marker_publisher.publish(self.drone_marker)

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
        inputs = tf.placeholder(tf.float32, (None, None, self.num_inputs))
        num_depth_inputs = (
            self.image_width * self.image_height * self.sequence_length)
        position = tf.reshape(inputs[:, :, num_depth_inputs:], [
            -1, self.state_dim * self.sequence_length + self.goal_dim
        ])
        depth = tf.reshape(inputs[:, :, :num_depth_inputs], [
            -1, self.image_height, self.image_width, self.sequence_length
        ])
        depth = tf.contrib.layers.conv2d(
            depth,
            num_outputs=16,
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
            kernel_size=(5, 5),
            stride=(2, 2),
            weights_regularizer=tf.nn.l2_loss)
        depth = tf.contrib.layers.batch_norm(depth)
        depth = tf.nn.relu(depth)
        depth = tf.contrib.layers.flatten(depth)
        depth = tf.contrib.layers.fully_connected(
            depth, 16, activation_fn=None, weights_regularizer=tf.nn.l2_loss)
        depth = tf.contrib.layers.batch_norm(depth)
        depth = tf.nn.relu(depth)
        position = tf.contrib.layers.fully_connected(
            position, 16, activation_fn=None, weights_regularizer=tf.nn.l2_loss)
        position = tf.contrib.layers.batch_norm(position)
        position = tf.nn.relu(position)
        x = tf.concat([position, depth], axis=-1)
        x = tf.contrib.layers.fully_connected(
            x, 16, activation_fn=None, weights_regularizer=tf.nn.l2_loss)
        x = tf.contrib.layers.batch_norm(x)
        x = tf.nn.relu(x)
        actions = tf.contrib.layers.fully_connected(
            inputs=x, num_outputs=self.action_dim, activation_fn=tf.nn.sigmoid)
        return inputs, actions

    def create_critic_network(self, scope):
        inputs = tf.placeholder(tf.float32, (None, None, self.num_inputs))
        actions = tf.placeholder(tf.float32, (None, None, self.action_dim))
        num_depth_inputs = (
            self.image_width * self.image_height * self.sequence_length)
        position = tf.reshape(inputs[:, :, num_depth_inputs:], [
            -1, self.state_dim * self.sequence_length + self.goal_dim
        ])
        depth = tf.reshape(inputs[:, :, :num_depth_inputs], [
            -1, self.image_height, self.image_width, self.sequence_length
        ])
        depth = tf.contrib.layers.conv2d(
            depth,
            num_outputs=16,
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
            kernel_size=(5, 5),
            stride=(2, 2),
            weights_regularizer=tf.nn.l2_loss)
        depth = tf.contrib.layers.batch_norm(depth)
        depth = tf.nn.relu(depth)
        depth = tf.contrib.layers.flatten(depth)
        depth = tf.contrib.layers.fully_connected(
            depth, 16, activation_fn=None, weights_regularizer=tf.nn.l2_loss)
        depth = tf.contrib.layers.batch_norm(depth)
        depth = tf.nn.relu(depth)
        position = tf.contrib.layers.fully_connected(
            position, 16, activation_fn=None, weights_regularizer=tf.nn.l2_loss)
        position = tf.contrib.layers.batch_norm(position)
        position = tf.nn.relu(position)
        act = tf.reshape(actions, [-1, self.action_dim])
        act = tf.contrib.layers.fully_connected(
            act, 16, activation_fn=None, weights_regularizer=tf.nn.l2_loss)
        act = tf.contrib.layers.batch_norm(act)
        act = tf.nn.relu(act)
        x = tf.concat([position, depth, act], axis=-1)
        x = tf.contrib.layers.fully_connected(
            x, 16, activation_fn=None, weights_regularizer=tf.nn.l2_loss)
        x = tf.contrib.layers.batch_norm(x)
        x = tf.nn.relu(x)
        out = tf.contrib.layers.fully_connected(
            inputs=x, num_outputs=1, activation_fn=None)
        return inputs, actions, out

    def get_current_frame(self):
        while not self.freshPose:
            print("Waiting for fresh pose...")
            rospy.sleep(0.05)

        self.freshPose = False
        position = np.array(
            [self.pose.position.x, self.pose.position.y, self.pose.position.z])
        depth_data = ros_numpy.numpify(self.depth_msg)
        depth_data[np.isnan(depth_data)] = 0
        # print(depth_data.shape)
        # plt.imshow(depth_data)
        # plt.show()
        # exit()

        depth = scipy.misc.imresize(
            depth_data, [self.image_height, self.image_width],
            mode='F').flatten()
        frame = (depth, position)
        return frame

    def get_current_state(self):
        frame = self.get_current_frame()
        self.frame_buffer.append(frame)
        while len(self.frame_buffer) < self.frame_buffer.maxlen:
            frame = self.get_current_frame()
            self.frame_buffer.append(frame)
        depth = np.concatenate(
            [frame[0] for frame in self.frame_buffer], axis=-1)
        position = np.concatenate(
            [frame[1] for frame in self.frame_buffer], axis=-1)
        state = np.concatenate([depth, position], axis=-1)
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
        except rospy.ServiceException:
            print("Failed to reset simulator.")

        # Reset our localization.
        com_msg = String()
        com_msg.data = "f reset"
        self.com_publisher.publish(com_msg)

        # Take-off.
        self.takeoff_publisher.publish(EmptyMessage())

        # bounds = 2
        # xy = (np.random.uniform(size=(2)) - 0.5) * (2 * bounds)
        # new_goal = np.array([xy[0], xy[1], 0])
        new_goal = [0, 4, 0]
        # print("New goal: %s" % new_goal)
        self.goal_pose.position.x = new_goal[0]
        self.goal_pose.position.y = new_goal[1]
        self.goal_pose.position.z = new_goal[2]
        self.goal_marker.pose.position.x = self.goal_pose.position.x
        self.goal_marker.pose.position.y = self.goal_pose.position.y
        self.goal_marker.pose.position.z = self.goal_pose.position.z
        self.goal_marker.action = Marker.MODIFY
        self.marker_publisher.publish(self.goal_marker)
        goal = np.array([
            self.goal_pose.position.x, self.goal_pose.position.y,
            self.goal_pose.position.z
        ])

        # Clear our frame buffer.
        self.frame_buffer.clear()

        state = self.get_current_state()
        return (state, goal)

    def step(self, state, action, goal):
        vel_msg = Twist()
        left_prob = action[0]
        right_prob = action[1]
        straight_prob = action[2]

        alpha = 0.5
        beta = 1.0
        if straight_prob > alpha:
            vel_msg.linear.x = beta
            vel_msg.angular.z = (right_prob - left_prob) * 2
            pass
        else:
            vel_msg.linear.x = 0.0
            if right_prob > left_prob:
                vel_msg.angular.z = beta
            else:
                vel_msg.angular.z = beta
            vel_msg.angular.z = (right_prob - left_prob) * 2

        # vel_msg.linear.x = action[0]
        # vel_msg.linear.y = 0
        # vel_msg.linear.z = 0
        # vel_msg.angular.z = action[1]
        self.velocity_publisher.publish(vel_msg)

        # Wait.
        self.rate.sleep()

        next_state = self.get_current_state()
        return next_state

    def reward(self, state, action, goal):
        position = state[-3:]
        distance = np.linalg.norm(position - goal)
        distance_reward = -(distance * distance)
        forward_reward = action[0]
        collided_reward = -1 if self.collided else 0
        reward_weights = np.array([1.0, 0.0, 0.0])
        reward = np.array([distance_reward, forward_reward, collided_reward])
        return np.dot(reward_weights, reward)

    def RunModel(self, model_name, num_attempts):
        env = Environment(self.reset, self.step, self.reward)
        actor_noise = OrnsteinUhlenbeckActionNoise(
            mu=np.zeros(self.num_actions))
        modeldir = os.path.join(
            os.path.dirname(__file__),
            '../../../learning/deep_drone/' + model_name)
        self.ddpg.RunModel(
            env, modeldir, actor_noise=None, num_attempts=num_attempts)

    def Train(self, prev_model):
        env = Environment(self.reset, self.step, self.reward)
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
            env, logdir=logdir, actor_noise=None, model_dir=modeldir)


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
