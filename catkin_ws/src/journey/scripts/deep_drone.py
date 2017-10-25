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
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
from tum_ardrone.msg import filter_state
from journey.srv import FlyToGoal
from journey.srv import FlyToGoalResponse

RATE = 10  # Hz


class DeepDronePlanner:

    def __init__(self):

        rospy.init_node('deep_drone_planner', anonymous=True)
        self.velocity_publisher = rospy.Publisher(
            '/cmd_vel', Twist, queue_size=10)

        # TODO(kirmani): Query the depth image instead of the RGB image.
        self.image_subscriber = rospy.Subscriber('/ardrone/front/image_raw',
                                                 Image, self._OnNewImage)
        self.pose_subscriber = rospy.Subscriber('/ardrone/predictedPose',
                                                filter_state, self._OnNewPose)

        s = rospy.Service('fly_to_goal', FlyToGoal, self.FlyToGoal)
        self.pose = Pose()
        self.rate = rospy.Rate(RATE)
        self.image_msg = None

        # Initialize goal
        self.goal_pose = Pose()
        self.goal_pose.position.x = 0
        self.goal_pose.position.y = 0
        self.goal_pose.position.z = 1

        # Actions.
        options = [-1, 0, 1]
        num_actions = 4
        self.actions = np.zeros((len(options)**num_actions, num_actions))
        for i in range(len(options)**num_actions):
            temp = i
            for j in range(num_actions):
                self.actions[i][j] = options[temp % len(options)]
                temp /= len(options)

        # Create model.
        self._CreateModel()

        # Start tensorflow session.
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # Initialize policy.
        self._InitializePolicy()

        print("Deep drone planner initialized.")

    def _CreateModel(self):
        self.image = tf.placeholder(
            tf.float32, (None, 360, 640, 3), name='input')
        self.delta = tf.placeholder(tf.float32, (None, 3), name='delta')
        self.actions = tf.placeholder(tf.float32, (None, 4), name='actions')
        self.reward = tf.placeholder(tf.float32, (None), name='reward')
        x = self.delta
        x = tf.contrib.layers.fully_connected(x, 32)
        x = tf.contrib.layers.fully_connected(x, 32)
        x = tf.contrib.layers.fully_connected(x, 4, activation_fn=None)
        self.controls = tf.clip_by_value(x, -1, 1)

        # Define the loss function
        self.loss = tf.reduce_mean(tf.abs(self.actions - x))
        # loss = tf.log(tf.nn.softmax(x)) * reward

        # Adam will likely converge much faster than SGD for this assignment.
        optimizer = tf.train.AdamOptimizer(0.001, 0.9, 0.999)

        # use that optimizer on your loss function (control_dependencies makes sure any
        # batch_norm parameters are properly updated)
        with tf.control_dependencies(
                tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optimizer = optimizer.minimize(self.loss)

    def _InitializePolicy(self):
        num_samples = 1000
        bounds = 5
        tolerance = 0.5
        delta = (np.random.uniform(size=(num_samples, 3)) - 0.5) * (2 * bounds)
        actions = np.zeros((num_samples, 4))
        actions[np.where(delta > tolerance)] = 1
        actions[np.where(delta < -tolerance)] = -1
        # print(delta)
        # print(actions)
        for epoch in range(300):
            loss_val, _ = self.sess.run(
                [self.loss, self.optimizer],
                feed_dict={self.delta: delta,
                           self.actions: actions})
        print("Policy initialization loss: %s" % loss_val)

    def _OnNewPose(self, data):
        self.pose.position.x = round(data.x, 4)
        self.pose.position.y = round(data.y, 4)
        self.pose.position.z = round(data.z, 4)

    def FlyToGoal(self, req):
        self.goal_pose.position.x = self.pose.position.x + req.x
        self.goal_pose.position.y = self.pose.position.y + req.y
        self.goal_pose.position.z = self.pose.position.z + req.z
        print("Flying to: (%s, %s, %s)" %
              (self.goal_pose.position.x, self.goal_pose.position.y,
               self.goal_pose.position.z))
        return FlyToGoalResponse(True)

    def _OnNewImage(self, image):
        self.image_msg = image

    def QueryPolicy(self, image, delta):
        # TODO(kirmani): Do on-policy learning here.
        image = np.stack([image])
        delta = np.stack([delta])
        controls = self.sess.run(
            [self.controls], feed_dict={self.image: image,
                                        self.delta: delta})[0][0]
        return controls

    def ImprovePolicy(self, delta, reward):
        # TODO(kirmani): Do policy optimization step.
        delta = np.stack([delta])
        self.sess.run(
            [self.optimizer],
            feed_dict={self.delta_input: delta,
                       self.reward_input: reward})

    def Plan(self):
        # Initialize velocity message.
        vel_msg = Twist()

        while not rospy.is_shutdown():
            if not self.image_msg:
                print("No image available.")
            else:
                # Create inputs for network.
                input_image = ros_numpy.numpify(self.image_msg)
                goal = np.array([
                    self.goal_pose.position.x, self.goal_pose.position.y,
                    self.goal_pose.position.z
                ])
                x = np.array([
                    self.pose.position.x, self.pose.position.y,
                    self.pose.position.z
                ])
                input_delta = goal - x

                # Output some control.
                controls = self.QueryPolicy(input_image, input_delta)
                print("Controls: %s" % controls)
                vel_msg.linear.x = controls[0]
                vel_msg.linear.y = controls[1]
                vel_msg.linear.z = controls[2]
                vel_msg.angular.z = controls[3]
                self.velocity_publisher.publish(vel_msg)

                # Wait.
                self.rate.sleep()

                # Distance after action.
                x = np.array([
                    self.pose.position.x, self.pose.position.y,
                    self.pose.position.z
                ])
                distance = np.linalg.norm(goal - x)

                # Get reward.
                reward = np.exp(-distance)
                print("Reward: %s" % reward)

                # # Improve policy.
                # self.ImprovePolicy(input_delta, reward)

                # # Start training on new goal if we succeed at this one.
                # if (reward > 0.95):
                #     print("Succeeded at reaching goal: %s" % goal)
                #     new_goal = np.random.uniform(size=3) * 3 + np.array(
                #         [0, 0, 1])
                #     self.goal_pose.position.x = new_goal[0]
                #     self.goal_pose.position.y = new_goal[1]
                #     self.goal_pose.position.z = new_goal[2]
                #     print("Going to new goal: %s" % new_goal)

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
