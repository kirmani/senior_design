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
                                                filter_state, self.callback)
        s = rospy.Service('fly_to_goal', FlyToGoal, self.fly_to_goal)
        self.pose = Pose()
        self.velocity = Pose()
        self.rate = rospy.Rate(RATE)

    #Callback function implementing the pose value received
    def callback(self, data):
        self.pose.position.x = round(data.x, 4)
        self.pose.position.y = round(data.y, 4)
        self.pose.position.z = round(data.z, 4)
        self.velocity.position.x = round(data.dx, 4)
        self.velocity.position.y = round(data.dy, 4)
        self.velocity.position.z = round(data.dz, 4)

    def fly_to_goal(self, req):
        if (req.x == 0 and req.y == 0 and req.z == 0):
            print("Landing.")
            self.land_publisher.publish()
            return FlyToGoalResponse(True)

        start = np.array(
            [self.pose.position.x, self.pose.position.y, self.pose.position.z])
        goal = np.array([start[0] + req.x, start[1] + req.y, start[2] + req.z])
        vel_msg = Twist()

        print("Flying to: (%s, %s, %s)" % (goal[0], goal[1], goal[2]))

        s = 1.0
        time = 5  # seconds
        alpha = -np.log(0.01)
        for i in range(time * RATE):
            x = np.array([
                self.pose.position.x, self.pose.position.y, self.pose.position.z
            ])
            distance = np.linalg.norm(goal - x)
            print("Distance: %s" % distance)
            v = np.array([
                self.velocity.position.x, self.velocity.position.y,
                self.velocity.position.z
            ])

            v_dot = ((goal - x) - v - (goal - start) * s) / time
            a = v + v_dot
            vel_msg.linear.x = a[0]
            vel_msg.linear.y = a[1]
            vel_msg.linear.z = a[1]
            s_dot = (-alpha / time) * s
            s += s_dot

            self.velocity_publisher.publish(vel_msg)
            self.rate.sleep()
        print(s)

        return FlyToGoalResponse(True)

    def Plan(self):
        vel_msg = Twist()
        self.velocity_publisher.publish(vel_msg)
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
