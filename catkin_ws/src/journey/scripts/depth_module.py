#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Sean Kirmani <sean@kirmani.io>
#
# Distributed under terms of the MIT license.
"""
Depth from RGB neural network module.
"""

import numpy as np
import os
import rospy
import ros_numpy
import tensorflow as tf
from depth_from_rgb import models
from matplotlib import pyplot as plt
from std_msgs.msg import String
from sensor_msgs.msg import Image

HEIGHT = 360
WIDTH = 640
CHANNELS = 3
BATCH_SIZE = 1
MODEL_DATA_PATH = '../tensorflow/models/depth_from_rgb/NYU_FCRN.ckpt'
DEPTH_MAX = 4


class DepthFromRGBNode:

    def __init__(self):
        # Create a placeholder for the input image
        self.input_node = tf.placeholder(
            tf.float32, shape=(None, HEIGHT, WIDTH, CHANNELS))

        # Construct the network
        self.net = models.ResNet50UpProj({
            'data': self.input_node
        }, BATCH_SIZE, 1, False)

        self.sess = tf.Session()

        # Load the converted parameters
        print('Loading the model')

        # Use to load from ckpt file
        saver = tf.train.Saver()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        saver.restore(self.sess, os.path.join(dir_path, MODEL_DATA_PATH))

        # Initialize ROS node.
        rospy.init_node('image_depth', anonymous=True)

        # Subscribe to camera feed.
        rospy.Subscriber('ardrone/front/image_raw', Image, self.on_new_image)

        # Create depth publisher.
        self.depth_publisher = rospy.Publisher(
            'ardrone/front/depth/image_raw', Image, queue_size=10)

        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()

    def on_new_image(self, data):
        rgb = ros_numpy.numpify(data)
        depth = (self.predict(np.expand_dims(rgb, axis=0))[0] * 256.0 /
                 DEPTH_MAX).astype(np.uint8)
        msg = ros_numpy.msgify(Image, depth, encoding='mono8')
        self.depth_publisher.publish(msg)

    def predict(self, img):
        prediction = self.sess.run(
            self.net.get_output(), feed_dict={self.input_node: img})
        return prediction


def main():
    depth_from_rgb = DepthFromRGBNode()


if __name__ == '__main__':
    main()
