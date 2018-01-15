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

import numpy as np
import rospy
import tf
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
        # rospy.sleep(5.)
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
        self.yaw_integral = 0.0
        self.yaw_prior = 0.0
        self.forward_integral = 0.0
        self.forward_prior = 0.0
        self.right_integral = 0.0
        self.right_prior = 0.0
        self.up_integral = 0.0
        self.up_prior = 0.0

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

                distance = np.linalg.norm(np.array([self.marker_pose.position.x, self.marker_pose.position.y]))
                angle = np.arctan2(-self.marker_pose.position.x, -self.marker_pose.position.y)
                quaternion = (self.marker_pose.orientation.x, self.marker_pose.orientation.y,
                              self.marker_pose.orientation.z, self.marker_pose.orientation.w)
                _, _, yaw = tf.transformations.euler_from_quaternion(quaternion)

                # Angular velocity in the XY plane.
                yaw_error = -yaw
                self.yaw_integral += yaw_error / self.update_rate
                yaw_derivative = (yaw_error - self.yaw_prior) * self.update_rate
                vel_msg.angular.z = np.clip(
                    kp * yaw_error + ki * self.yaw_integral +
                    kd * yaw_derivative, -1, 1)
                self.yaw_prior = yaw_error
                print(yaw_error)

                # Linear velocity in the forward axis
                forward_error = distance * np.cos(angle)
                self.forward_integral += forward_error / self.update_rate
                forward_derivative = (
                    forward_error - self.forward_prior) * self.update_rate
                vel_msg.linear.x = np.clip(kp * forward_error +
                                           ki * self.forward_integral +
                                           kd * forward_derivative, -1,
                                           1)
                self.forward_prior = forward_error

                # Linear velocity in right axis.
                right_error = distance * np.sin(angle)
                self.right_integral += right_error / self.update_rate
                right_derivative = (
                    right_error - self.right_prior) * self.update_rate
                vel_msg.linear.y = np.clip(kp * right_error +
                                           ki * self.right_integral +
                                           kd * right_derivative, -1,
                                           1)
                self.right_prior = right_error

                # Up axis.
                up_error = self.hover_altitude - self.marker_pose.position.z
                self.up_integral += up_error / self.update_rate
                up_derivative = (up_error - self.up_prior) * self.update_rate
                vel_msg.linear.z = np.clip(
                    kp * up_error + ki * self.up_integral +
                    kd * up_derivative, -1, 1)
                self.up_prior = up_error

            # Publish our velocity message.
            self.velocity_publisher.publish(vel_msg)

            # Wait.
            self.rate.sleep()


def main():
    tag_tracker = TagTrackerNode()


if __name__ == '__main__':
    main()
