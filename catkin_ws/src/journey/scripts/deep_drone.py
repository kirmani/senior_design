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


class ActorNetwork:

    def __init__(self,
                 sess,
                 num_inputs,
                 num_actions,
                 tau=0.1,
                 learning_rate=0.001):
        self.sess = sess
        self.inputs = tf.placeholder(tf.float32, (None, num_inputs))
        x = tf.contrib.layers.fully_connected(self.inputs, 32)
        x = tf.contrib.layers.fully_connected(x, 32)
        self.actions = tf.contrib.layers.fully_connected(
            x, num_actions, activation_fn=tf.tanh)

        network_params = tf.trainable_variables()

        target_network_params = tf.trainable_variables()[len(network_params):]

        # Op for periodically updating target network with online network weights
        update_target_network_params = \
            [target_network_params[i].assign(tf.mul(network_params[i], tau) + \
                tf.mul(target_network_params[i], 1. - tau))
                for i in range(len(target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, num_actions])

        # Combine the gradients here
        self.actor_gradients = tf.gradients(self.actions, network_params,
                                            -self.action_gradient)

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(learning_rate).\
            apply_gradients(zip(self.actor_gradients, network_params))

    def train(self, inputs, a_gradient):
        self.sess.run(
            self.optimize,
            feed_dict={self.inputs: inputs,
                       self.action_gradient: a_gradient})

    def predict(self, inputs):
        return self.sess.run(self.actions, feed_dict={self.inputs: inputs})


class CriticNetwork:

    def __init__(self, sess, num_inputs, num_actions, learning_rate=0.001):
        self.sess = sess
        self.inputs = tf.placeholder(tf.float32, (None, num_inputs))
        self.actions = tf.placeholder(tf.float32, (None, num_actions))
        x = tf.contrib.layers.fully_connected(self.inputs, 32)
        x = tf.concat([x, self.actions], axis=-1)
        x = tf.contrib.layers.fully_connected(x, 32)
        self.out = tf.contrib.layers.fully_connected(x, 1, activation_fn=None)

        # Network target (y_i)
        # Obtained from the target networks
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tf.losses.mean_squared_error(self.out,
                                                 self.predicted_q_value)
        self.optimize = tf.train.AdamOptimizer(learning_rate).minimize(
            self.loss)

        # Get the gradient of the net w.r.t. the action
        self.action_grads = tf.gradients(self.out, self.actions)

    def train(self, inputs, actions, reward):
        self.sess.run(
            self.optimize,
            feed_dict={
                self.inputs: inputs,
                self.actions: actions,
                self.predicted_q_value: reward
            })

    def predict(self, inputs, actions):
        return self.sess.run(
            self.out, feed_dict={self.inputs: inputs,
                                 self.actions: actions})


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

        # Start tensorflow session.
        sess = tf.Session()

        # Create actor network.
        self.actor = ActorNetwork(sess, 3, 4)
        self.critic = CriticNetwork(sess, 3, 4)

        sess.run(tf.global_variables_initializer())

        # Initialize policy.
        self._InitializePolicy()

        print("Deep drone planner initialized.")

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
            self.actor.train(delta, actions)
        # print("Policy initialization loss: %s" % loss_val)
        exit()

    def _OnNewPose(self, data):
        self.pose.position.x = round(data.x, 4)
        self.pose.position.y = round(data.y, 4)
        self.pose.position.z = round(data.z, 4)

    def _OnNewImage(self, image):
        self.image_msg = image

    def FlyToGoal(self, req):
        self.goal_pose.position.x = self.pose.position.x + req.x
        self.goal_pose.position.y = self.pose.position.y + req.y
        self.goal_pose.position.z = self.pose.position.z + req.z
        print("Flying to: (%s, %s, %s)" %
              (self.goal_pose.position.x, self.goal_pose.position.y,
               self.goal_pose.position.z))
        return FlyToGoalResponse(True)

    # def QueryPolicy(self, delta):
    #     # TODO(kirmani): Do on-policy learning here.
    #     delta = np.stack([delta])
    #     controls = self.sess.run(
    #         [self.controls], feed_dict={self.delta: delta})[0][0]
    #     return controls

    # def ImprovePolicy(self, delta, reward):
    #     # TODO(kirmani): Do policy optimization step.
    #     delta = np.stack([delta])
    #     reinforcement_loss, _ = self.sess.run(
    #         [self.reinforcement_loss, self.reinforcement_optimizer],
    #         feed_dict={self.delta: delta,
    #                    self.reward: reward})
    #     print("Reinforcement loss: %s" % reinforcement_loss)

    def Plan(self):
        # Initialize velocity message.
        vel_msg = Twist()

        while not rospy.is_shutdown():
            # if not self.image_msg:
            #     print("No image available.")
            # else:
            #     # Create inputs for network.
            #     image = ros_numpy.numpify(self.image_msg)
            goal = np.array([
                self.goal_pose.position.x, self.goal_pose.position.y,
                self.goal_pose.position.z
            ])
            x = np.array([
                self.pose.position.x, self.pose.position.y, self.pose.position.z
            ])
            delta = goal - x

            # Output some control.
            controls = self.actor.predict(delta)
            vel_msg.linear.x = controls[0]
            vel_msg.linear.y = controls[1]
            vel_msg.linear.z = controls[2]
            vel_msg.angular.z = controls[3]
            self.velocity_publisher.publish(vel_msg)
            # print("Controls: %s" % controls)

            # Wait.
            self.rate.sleep()

            # Distance after action.
            x = np.array([
                self.pose.position.x, self.pose.position.y, self.pose.position.z
            ])
            distance = np.linalg.norm(goal - x)

            # Get reward.
            reward = np.exp(-distance)
            # print("Reward: %s" % reward)

            # Improve policy.
            self.ImprovePolicy(delta, reward)

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
