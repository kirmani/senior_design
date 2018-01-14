#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Sean Kirmani <sean@kirmani.io>
#
# Distributed under terms of the MIT license.
"""
AR tag tracker.
"""

import rospy
import numpy as np
from ar_track_alvar_msgs.msg import AlvarMarkers
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty


class TagTrackerNode:

    def __init__(self, rate=4):
        # Initialize ROS node.
        rospy.init_node('tag_tracker', anonymous=True)

        # Nav command update frequency in Hz.
        self.update_rate = rate
        self.rate = rospy.Rate(self.update_rate)

        # Toggle camera to bottom.
        rospy.wait_for_service('/ardrone/togglecam')
        rospy.sleep(5.)
        toggle_cam = rospy.ServiceProxy('/ardrone/togglecam', Empty)
        toggle_cam()

        # Look for AR tags.
        rospy.Subscriber('/ar_pose_marker', AlvarMarkers,
            self.on_new_markers)
        self.marker_pose = Pose()
        self.marker_dirty = False

        # Actions.
        self.velocity_publisher = rospy.Publisher(
            '/cmd_vel', Twist, queue_size=10)

        # Desired altitude.
        self.hover_altitude = 0.5

        # Reset PID integrals and priors.
        self.x_integral = 0.0
        self.x_prior = 0.0
        self.y_integral = 0.0
        self.y_prior = 0.0
        self.z_integral = 0.0
        self.z_prior = 0.0

        print("Tag tracker initialzated.")

        # Begin planning loop.
        self.planning_loop()

    def on_new_markers(self, markers):
        for marker in markers.markers:
            if marker.id == 0:
                self.marker_pose = marker.pose.pose
                self.marker_dirty = True

    def planning_loop(self):
        # PID constants.
        kp = 0.6
        ki = 0.001
        kd = 0.1

        while not rospy.is_shutdown():
            vel_msg = Twist()

            if self.marker_dirty:
                self.marker_dirty = False

                # Forward axis.
                x_error = -self.marker_pose.position.y
                self.x_integral += x_error / self.update_rate
                x_derivative = (x_error - self.x_prior) * self.update_rate
                vel_msg.linear.x = np.clip(
                    kp * x_error + ki * self.x_integral +
                    kd * x_derivative, -1, 1)
                self.x_prior = x_error

                # Right axis.
                y_error = -self.marker_pose.position.x
                self.y_integral += y_error / self.update_rate
                y_derivative = (y_error - self.y_prior) * self.update_rate
                vel_msg.linear.y = np.clip(
                    kp * y_error + ki * self.y_integral +
                    kd * y_derivative, -1, 1)
                self.y_prior = y_error

                # Up axis.
                z_error = self.hover_altitude - self.marker_pose.position.z
                self.z_integral += z_error / self.update_rate
                z_derivative = (z_error - self.z_prior) * self.update_rate
                vel_msg.linear.z = np.clip(
                    kp * z_error + ki * self.z_integral +
                    kd * z_derivative, -1, 1)
                self.z_prior = z_error

            # Publish our velocity message.
            self.velocity_publisher.publish(vel_msg)

            # Wait.
            self.rate.sleep()


def main():
    tag_tracker = TagTrackerNode()


if __name__ == '__main__':
    main()
