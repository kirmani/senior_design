#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Sean Kirmani <sean@kirmani.io>
#
# Distributed under terms of the MIT license.
"""
ARDrone 2.0 Trajectory planner.
"""
import argparse
import numpy as np
import rospy
import sys
import traceback
import time
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from std_msgs.msg import Empty
from tum_ardrone.msg import filter_state
from journey.srv import FlyToGoal
from journey.srv import FlyToGoalResponse

GOAL_DISTANCE_THRESHOLD = 0.05
RATE = 10


class TrajectoryPlanner:

    def __init__(self):
        rospy.init_node('trajectory_planner', anonymous=True)
        self.takeoff_publisher = rospy.Publisher(
            '/ardrone/takeoff', Empty, queue_size=10)
        self.land_publisher = rospy.Publisher(
            '/ardrone/land', Empty, queue_size=10)
        self.velocity_publisher = rospy.Publisher(
            '/cmd_vel', Twist, queue_size=10)
        self.pose_subscriber = rospy.Subscriber('/ardrone/predictedPose',
                                                filter_state, self._OnNewPose)
        s = rospy.Service('fly_to_goal', FlyToGoal, self.FlyToGoal)
        self.pose = Pose()
        self.velocity = Pose()
        self.rate = rospy.Rate(RATE)

        # Start in hover mode.
        vel_msg = Twist()
        vel_msg.linear.x = 0
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        self.velocity_publisher.publish(vel_msg)

    def _OnNewPose(self, data):
        self.pose.position.x = round(data.x, 4)
        self.pose.position.y = round(data.y, 4)
        self.pose.position.z = round(data.z, 4)
        self.velocity.position.x = round(data.dx, 4)
        self.velocity.position.y = round(data.dy, 4)
        self.velocity.position.z = round(data.dz, 4)

    def FlyToGoal(self, req):
        start = np.array(
            [self.pose.position.x, self.pose.position.y, self.pose.position.z])
        delta = np.array([req.x, req.y, req.z])
        time = req.time
        goal = start + delta

        print("Flying to: (%s, %s, %s)" % (goal[0], goal[1], goal[2]))

        s = 1.0
        alpha = -np.log(0.01)
        vel_msg = Twist()
        for i in range(int(time * RATE)):
            # Query position and velocity.
            x = np.array([
                self.pose.position.x, self.pose.position.y, self.pose.position.z
            ])
            v = np.array([
                self.velocity.position.x, self.velocity.position.y,
                self.velocity.position.z
            ])
            distance = np.linalg.norm(goal - x)
            print("Distance: %s" % distance)

            # Update and publish velocity.
            v_dot = ((goal - x) - v - (goal - start) * s) / time
            a = v + v_dot
            vel_msg.linear.x = a[0]
            vel_msg.linear.y = a[1]
            vel_msg.linear.z = a[1]
            self.velocity_publisher.publish(vel_msg)

            # Update phase.
            s_dot = (-alpha / time) * s
            s += s_dot

            # Wait.
            self.rate.sleep()

        # Go back to hover mode when done.
        vel_msg.linear.x = 0
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        self.velocity_publisher.publish(vel_msg)

        return FlyToGoalResponse(True)

    def Plan(self):
        print("Trajectory planner initialized.")
        rospy.spin()


def main(args):
    """ Main function. """
    trajectory_planner = TrajectoryPlanner()
    trajectory_planner.Plan()


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
