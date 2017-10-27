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
import rospy
import ros_numpy
import os
import sys
import traceback
import time
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
from std_msgs.msg import Empty as EmptyMessage
from std_msgs.msg import String
from std_srvs.srv import Empty as EmptyService
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
        # TODO(kirmani): Query the depth image instead of the RGB image.
        self.image_subscriber = rospy.Subscriber('/ardrone/front/image_raw',
                                                 Image, self._OnNewImage)
        self.pose_subscriber = rospy.Subscriber('/ardrone/predictedPose',
                                                filter_state, self._OnNewPose)
        self.pose = Pose()
        self.image_msg = None

        # Actions.
        self.velocity_publisher = rospy.Publisher(
            '/cmd_vel', Twist, queue_size=10)

        # Reset topics.
        self.takeoff_publisher = rospy.Publisher(
            '/ardrone/takeoff', EmptyMessage, queue_size=10)
        self.com_publisher = rospy.Publisher(
            '/tum_ardrone/com', String, queue_size=10)

        # Listen for new goal when planning at test time.
        s = rospy.Service('fly_to_goal', FlyToGoal, self.FlyToGoal)

        # The rate which we publish commands.
        self.rate = rospy.Rate(self.rate)

        # Initialize goal.
        self.goal_pose = Pose()

        # Set up policy search network.
        self.num_inputs = 3
        self.num_actions = 4
        self.ddpg = DeepDeterministicPolicyGradients(self.num_inputs,
                                                     self.num_actions)

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
        # new_goal = (np.random.uniform(size=(3)) - 0.5) * (2 * bounds)
        # new_goal += np.array([0, 0, bounds])
        new_goal = [-2, 2, 1]
        # print("New goal: %s" % new_goal)
        self.goal_pose.position.x = new_goal[0]
        self.goal_pose.position.y = new_goal[1]
        self.goal_pose.position.z = new_goal[2]
        goal = np.array([
            self.goal_pose.position.x, self.goal_pose.position.y,
            self.goal_pose.position.z
        ])
        x = np.array(
            [self.pose.position.x, self.pose.position.y, self.pose.position.z])
        state = goal - x
        return state

    def step(self, action):
        vel_msg = Twist()
        vel_msg.linear.x = action[0]
        vel_msg.linear.y = action[1]
        vel_msg.linear.z = action[2]
        vel_msg.angular.z = action[3]
        self.velocity_publisher.publish(vel_msg)

        # Wait.
        self.rate.sleep()

        # Get next state.
        goal = np.array([
            self.goal_pose.position.x, self.goal_pose.position.y,
            self.goal_pose.position.z
        ])
        x = np.array(
            [self.pose.position.x, self.pose.position.y, self.pose.position.z])
        next_state = goal - x

        # Get reward.
        distance = np.linalg.norm(next_state)
        terminal = (distance < self.distance_threshold)
        if terminal:
            reward = 100
        else:
            reward = np.exp(-distance)

        return next_state, reward, terminal

    def Train(self):
        env = Environment(self.reset, self.step)
        actor_noise = OrnsteinUhlenbeckActionNoise(
            mu=np.zeros(self.num_actions))
        logdir = os.path.join(
            os.path.dirname(__file__), '../../../learning/deep_drone/')
        self.ddpg.Train(
            env,
            actor_noise,
            logdir=logdir,
            max_episodes=50,
            max_episode_len=30)


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
