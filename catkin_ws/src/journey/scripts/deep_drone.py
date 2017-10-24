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

        (self.image_input, self.control_output) = self._CreateModel()

        print("Deep drone planner initialized.")

    def _CreateModel(self):
        inputs = tf.placeholder(tf.float32, (None, 360, 640, 3), name='input')
        x = tf.contrib.layers.conv2d(
            inputs, 64, [3, 3], stride=2, scope="conv1")
        x = tf.contrib.layers.conv2d(x, 128, [3, 3], stride=2, scope="conv2")
        x = tf.contrib.layers.conv2d(x, 256, [3, 3], stride=2, scope="conv3")
        x = tf.contrib.layers.conv2d(x, 512, [3, 3], stride=2, scope="conv4")
        x = tf.contrib.layers.conv2d(x, 1024, [3, 3], stride=2, scope="conv5")
        x = tf.contrib.layers.fully_connected(x, 128)
        x = tf.contrib.layers.flatten(x)
        x = tf.contrib.layers.fully_connected(x, 4, activation_fn=None)
        outputs = tf.clip_by_value(x, -1, 1)
        return (inputs, outputs)

    def _OnNewImage(self, image):
        self.image = image

    def Reward(self, image):
        # TODO(kirmani): Design reward function.
        return 1

    def QueryPolicy(self, sess, image):
        # TODO(kirmani): Do on-policy learning here.
        image = np.stack([image])
        controls = sess.run(
            [self.control_output], feed_dict={self.image_input: image})[0][0]
        return controls

    def Plan(self):
        # Start tensorflow session.
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # Initialize velocity message.
        vel_msg = Twist()

        while not rospy.is_shutdown():
            if not self.image:
                print("No image available.")
            else:
                input_image = ros_numpy.numpify(self.image)

                controls = self.QueryPolicy(sess, input_image)
                print(controls)

                vel_msg.linear.x = controls[0]
                vel_msg.linear.y = controls[1]
                vel_msg.linear.z = controls[2]
                vel_msg.angular.z = controls[3]
                self.velocity_publisher.publish(vel_msg)

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
