#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Sean Kirmani <sean@kirmani.io>
#
# Distributed under terms of the MIT license.
"""
Navigation planner.
"""

import rospy
import numpy as np
import tf
from journey.msg import CollisionState
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty as EmptyMessage


class NavigationPlannerNode:

    def __init__(self, rate=4):
        # Initialize ROS node.
        rospy.init_node('nav_planner', anonymous=True)

        # Nav command update frequency in Hz.
        self.update_rate = rate
        self.rate = rospy.Rate(self.update_rate)

        # Pose subscriber.
        self.pose_subscriber = rospy.Subscriber('/ground_truth/state', Odometry,
                                                self.on_new_pose)
        self.pose = Pose()

        # Actions.
        self.velocity_publisher = rospy.Publisher(
            '/cmd_vel', Twist, queue_size=10)

        # Subscribe to collision state.
        self.collision_state_subscriber = rospy.Subscriber(
            '/ardrone/collision_state', CollisionState,
            self.on_new_collision_state)
        self.collision_state = None

        # How much to consider collision evidence in our nav planner relative
        # to our PID control for the goal.
        self.collision_weight = 0.1

        # Takeoff publisher.
        self.takeoff_publisher = rospy.Publisher(
            '/ardrone/takeoff', EmptyMessage, queue_size=10)

        # Set navigation goal service.
        self.set_nav_goal_service = rospy.Subscriber('/journey/set_nav_goal',
                                                     Point, self.set_nav_goal)
        self.nav_goal = Pose()

        # Take-off drone.
        rospy.sleep(2.0)
        self.set_nav_goal(Point(0.0, 0.0, 1.0))
        self.takeoff_publisher.publish(EmptyMessage())

        print("Navigation planner initialzated.")

        # Begin planning loop.
        self.planning_loop()

    def on_new_pose(self, state):
        self.pose = state.pose.pose

    def on_new_collision_state(self, collision_state):
        self.collision_state = collision_state

    def set_nav_goal(self, nav_delta):
        self.nav_goal.position.x = self.pose.position.x + nav_delta.x
        self.nav_goal.position.y = self.pose.position.y + nav_delta.y
        self.nav_goal.position.z = nav_delta.z
        print("Set navigation goal: (%.4f, %.4f, %.4f)" %
              (self.nav_goal.position.x, self.nav_goal.position.y,
               self.nav_goal.position.z))

    def planning_loop(self):
        # Velocity control scaling constant.
        forward_kp = 0.2

        # Gaz PID variables.
        up_kp = 0.2
        up_ki = 0.0
        up_kd = 0.0
        up_integral = 0.0
        up_prior = 0.0

        # Yaw PID variables.
        yaw_kp = -1.0
        yaw_ki = 0.0
        yaw_kd = 0.0
        yaw_integral = 0.0
        yaw_prior = 0.0

        # Tolerance around nav goal in meters.
        distance_threshold = 0.5

        while not rospy.is_shutdown():
            x = np.array([
                self.pose.position.x, self.pose.position.y, self.pose.position.z
            ])
            quaternion = (self.pose.orientation.x, self.pose.orientation.y,
                          self.pose.orientation.z, self.pose.orientation.w)
            _, _, yaw = tf.transformations.euler_from_quaternion(quaternion)
            g = np.array([
                self.nav_goal.position.x, self.nav_goal.position.y,
                self.nav_goal.position.z
            ])

            # Proportional controller.
            vel_msg = Twist()

            distance = np.linalg.norm(g[:2] - x[:2])
            if distance > distance_threshold:
                # Angular velocity in the XY plane.
                angle = np.arctan2(g[1] - x[1], g[0] - x[0])
                yaw_error = angle - yaw
                yaw_integral += yaw_error / self.update_rate
                yaw_derivative = (yaw_error - yaw_prior) * self.update_rate
                vel_msg.angular.z = np.clip(
                    yaw_kp * yaw_error + yaw_ki * yaw_integral +
                    yaw_kd * yaw_derivative, -1, 1)
                yaw_prior = yaw_error

                # Linear velocity in the forward axis.
                vel_msg.linear.x = np.clip(
                    forward_kp * distance * np.cos(angle, yaw), -1, 1)

                # Only weight obstacle avoidance if we've received information
                # from our collision avoidance network.
                # if self.collision_state:
                #     # Factor in our collision information into our navigation
                #     # plan.
                #     vel_msg.linear.x = (
                #         self.collision_weight * self.collision_state.action[0] +
                #         (1 - self.collision_weight) * vel_msg.linear.x)
                #     vel_msg.angular.z = (
                #         self.collision_weight * self.collision_state.action[1] +
                #         (1 - self.collision_weight) * vel_msg.angular.z)

            # Linear velocity in the up axis.
            up_error = np.linalg.norm(g[2] - x[2])
            up_integral += up_error / self.update_rate
            up_derivative = (up_error - up_prior) * self.update_rate
            vel_msg.linear.z = np.clip(
                up_kp * up_error + up_ki * up_integral + up_kd * up_derivative,
                -1, 1)
            up_prior = up_error

            # Publish our velocity message.
            self.velocity_publisher.publish(vel_msg)

            # Wait.
            self.rate.sleep()


def main():
    nav_planner = NavigationPlannerNode()


if __name__ == '__main__':
    main()
