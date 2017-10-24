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
import sys
import tensorflow as tf
import traceback
import time
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

RATE = 10  # Hz


class DeepDronePlanner:

    def __init__(self):
        rospy.init_node('deep_drone_planner', anonymous=True)
        self.velocity_publisher = rospy.Publisher(
            '/cmd_vel', Twist, queue_size=10)
        self.pose_subscriber = rospy.Subscriber('/ardrone/front/image_raw',
                                                Image, self._OnNewImage)
        self.rate = rospy.Rate(RATE)
        self.image = None

        print("Trajectory planner initialized.")

    def _OnNewImage(self, image):
        self.image = image

    def QueryPolicy(self, image):
        # TODO(kirmani): Do on-policy learning here.
        return [0, 0, 0, 0]

    def Plan(self):
        vel_msg = Twist()
        while not rospy.is_shutdown():
            if not self.image:
                print("No image available.")
            else:
                input_image = ros_numpy.numpify(self.image)

                controls = self.QueryPolicy(input_image)

                vel_msg.linear.x = controls[0]
                vel_msg.linear.y = controls[1]
                vel_msg.linear.z = controls[2]
                vel_msg.angular.z = controls[3]

            # Wait.
            self.rate.sleep()


def main(args):
    deep_drone_planner = DeepDronePlanner()
    deep_drone_planner.Plan()


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
