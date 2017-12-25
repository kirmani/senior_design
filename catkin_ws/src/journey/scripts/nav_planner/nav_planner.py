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

    def set_nav_goal(self, nav_delta):
        self.nav_goal.position.x = self.pose.position.x + nav_delta.x
        self.nav_goal.position.y = self.pose.position.y + nav_delta.y
        self.nav_goal.position.z = self.pose.position.z + nav_delta.z
        print("Set navigation goal: (%.4f, %.4f, %.4f)" %
              (self.nav_goal.position.x, self.nav_goal.position.y,
               self.nav_goal.position.z))

    def planning_loop(self):
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

            if np.linalg.norm(g[:2] - x[:2]) > 0.5:
                # Linear velocity in the forward axis.
                vel_msg.linear.x = np.clip(0.2 * np.linalg.norm(g[:2] - x[:2]),
                                           -1, 1)

                # Angular velocity in the XY plane.
                vel_msg.angular.z = np.clip(
                    -4 * (np.arctan2(g[1] - x[1], g[0] - x[0]) - yaw), -1, 1)

            # Linear velocity in the up axis.
            vel_msg.linear.z = np.clip(0.2 * np.linalg.norm(g[2] - x[2]), -1, 1)

            # Publish our velocity message.
            self.velocity_publisher.publish(vel_msg)

            # Wait.
            self.rate.sleep()


def main():
    nav_planner = NavigationPlannerNode()


if __name__ == '__main__':
    main()
