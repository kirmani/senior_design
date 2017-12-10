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
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from journey.srv import FlyToGoal
from std_msgs.msg import Empty as EmptyMessage
from tum_ardrone.msg import filter_state


class NavigationPlannerNode:

    def __init__(self, rate=4):
        # Initialize ROS node.
        rospy.init_node('nav_planner', anonymous=True)

        # Nav command update frequency in Hz.
        self.rate = rospy.Rate(rate)

        # Pose subscriber.
        self.pose_subscriber = rospy.Subscriber(
            '/ardrone/predictedPose', filter_state, self.on_new_pose)
        self.pose = Pose()

        # Actions.
        self.velocity_publisher = rospy.Publisher(
            '/cmd_vel', Twist, queue_size=10)

        # Takeoff publisher.
        self.takeoff_publisher = rospy.Publisher(
            '/ardrone/takeoff', EmptyMessage, queue_size=10)

        # Set navigation goal service.
        self.set_nav_goal_service = rospy.Service('set_nav_goal', FlyToGoal, self.set_nav_goal)
        self.nav_goal = Pose()
        self.nav_goal.position.z = 1.0

        # Take-off drone.
        rospy.sleep(2.0)
        self.takeoff_publisher.publish(EmptyMessage())

        print("Navigation planner initialzated.")

        # Begin planning loop.
        self.planning_loop()

    def on_new_pose(self, pose):
        self.pose.position.x = pose.x
        self.pose.position.y = pose.y
        self.pose.position.z = pose.z

    def set_nav_goal(self, nav_goal):
        self.nav_goal.position.x = self.pose.position.x + nav_goal.dx
        self.nav_goal.position.y = self.pose.position.y + nav_goal.dy
        self.nav_goal.position.z = self.pose.position.z + nav_goal.dz

    def planning_loop(self):
        while not rospy.is_shutdown():
            x = np.array([
                    self.pose.position.x,
                    self.pose.position.y,
                    self.pose.position.z
                ])
            g = np.array([
                    self.nav_goal.position.x,
                    self.nav_goal.position.y,
                    self.nav_goal.position.z
                    ])
            kp = 0.1
            delta = kp * (g - x)

            vel_msg = Twist()
            vel_msg.linear.x = delta[0]
            vel_msg.linear.y = delta[1]
            vel_msg.linear.z = delta[2]
            vel_msg.angular.z = 0
            self.velocity_publisher.publish(vel_msg)

            # Wait.
            self.rate.sleep()



def main():
    nav_planner = NavigationPlannerNode()


if __name__ == '__main__':
    main()
