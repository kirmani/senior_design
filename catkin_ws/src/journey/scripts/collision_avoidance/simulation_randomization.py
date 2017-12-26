#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Sean Kirmani <sean@kirmani.io>
#
# Distributed under terms of the MIT license.
"""
Simulation randomization.

When called, our simulation environment resets to some random configuration
to start training a new episode.

If the model sees enough simulated variation, the real world may look just like
the next simulator.
"""
import argparse
import numpy as np
import rospy
import random
import sys
import traceback
import time
import tf
from gazebo_msgs.srv import DeleteModel
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose

MATERIALS = [
    'Gazebo/White',
    'Gazebo/Black',
    'Gazebo/Grey',
    'Gazebo/Blue',
    'Gazebo/Red',
    'Gazebo/Green',
    'Gazebo/Purple',
    'Gazebo/Yellow',
    'Gazebo/Turquoise',
    'Gazebo/Grey',
    'Gazebo/WoodFloor',
    'Gazebo/CeilingTiled',
    'Gazebo/PaintedWall',
    'Gazebo/CloudySky'
    'Gazebo/GrassFloor',
    'Gazebo/Rockwall',
    'Gazebo/RustyBarrel',
    'Gazebo/WoodPallet',
    'Gazebo/LightWood'
    'Gazebo/WoodTile',
    'Gazebo/Brick',
    'Gazebo/Gold',
    'Gazebo/RustySteel',
    'Gazebo/Chrome',
    'Gazebo/BumpyMetal',
    'Gazebo/Rocky',
]


class SimulationRandomizer:

    def __init__(self):
        self.min_hallway_width = 2.0
        self.max_hallway_width = 4.0
        self.min_wall_height = 2.0
        self.max_wall_height = 5.0
        self.wall_width = 0.1
        self.wall_length = 100.0
        self.quadrotor_width = 0.5
        self.max_quadrotor_start_yaw = 45  # degrees

        # Publish model state.
        self.model_state_publisher = rospy.Publisher(
            '/gazebo/set_model_state', ModelState, queue_size=10)

        print("Initialized simulation randomizer.")

    def __call__(self):
        print("Randomized simulation.")

        # Pick randomized parameters.
        hallway_width = (np.random.random() *
                         (self.max_hallway_width - self.min_hallway_width) +
                         self.min_hallway_width)
        wall_height = (np.random.random() *
                       (self.max_wall_height - self.min_wall_height) +
                       self.min_wall_height)
        wall_length = self.wall_length
        wall_width = self.wall_width
        quadrotor_ty = (np.random.random() *
                        (hallway_width - self.quadrotor_width) -
                        (hallway_width - self.quadrotor_width) / 2)
        quadrotor_yaw = (2.0 * np.random.random() * self.max_quadrotor_start_yaw
                         - self.max_quadrotor_start_yaw) * np.pi / 180.0

        # Delete models.
        rospy.wait_for_service('gazebo/delete_model')
        delete_model = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
        delete_model('left_wall')
        delete_model('right_wall')
        delete_model('floor')
        delete_model('ceiling')
        delete_model('deadend_front')
        delete_model('deadend_back')

        # Create left wall.
        self.spawn_box(
            model_name='left_wall',
            ty=(-(hallway_width + wall_width) / 2),
            tz=(wall_height / 2),
            sx=wall_length,
            sy=wall_width,
            sz=wall_height,
            material=random.choice(MATERIALS))

        # Create right wall.
        self.spawn_box(
            model_name='right_wall',
            ty=((hallway_width + wall_width) / 2),
            tz=(wall_height / 2),
            sx=wall_length,
            sy=wall_width,
            sz=wall_height,
            material=random.choice(MATERIALS))

        # Create floor.
        self.spawn_box(
            model_name='floor',
            tz=(-wall_width / 2),
            sx=wall_length,
            sy=hallway_width,
            sz=wall_width,
            material=random.choice(MATERIALS))

        # Create ceiling.
        self.spawn_box(
            model_name='ceiling',
            tz=(wall_height + wall_width / 2),
            sx=wall_length,
            sy=hallway_width,
            sz=wall_width,
            material=random.choice(MATERIALS))

        # Create dead-end (front).
        self.spawn_box(
            model_name='deadend_front',
            tx=(wall_length / 2),
            tz=(wall_height / 2),
            sx=wall_width,
            sy=hallway_width,
            sz=wall_height,
            material=random.choice(MATERIALS))

        # Create dead-end (back).
        self.spawn_box(
            model_name='deadend_back',
            tx=(-wall_length / 2),
            tz=(wall_height / 2),
            sx=wall_width,
            sy=hallway_width,
            sz=wall_height,
            material=random.choice(MATERIALS))

        self.spawn_quadrotor(ty=quadrotor_ty, yaw=quadrotor_yaw)

        # Wait a little bit for environment to stabilize.
        rospy.sleep(2.)

    def spawn_quadrotor(self, tx=0, ty=0, tz=1, roll=0, pitch=0, yaw=0):
        position = (tx, ty, ty)
        quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)

        reset_pose = Pose()
        reset_pose.position.x = position[0]
        reset_pose.position.y = position[1]
        reset_pose.position.z = position[2]
        reset_pose.orientation.x = quaternion[0]
        reset_pose.orientation.y = quaternion[1]
        reset_pose.orientation.z = quaternion[2]
        reset_pose.orientation.w = quaternion[3]

        model_state = ModelState()
        model_state.model_name = 'quadrotor'
        model_state.reference_frame = 'world'
        model_state.pose = reset_pose
        self.model_state_publisher.publish(model_state)

    def spawn_box(self,
                  model_name="box",
                  tx=0,
                  ty=0,
                  tz=0.5,
                  yaw=0,
                  pitch=0,
                  roll=0,
                  sx=1,
                  sy=1,
                  sz=1,
                  static=True,
                  material='Gazebo/Blue'):
        s = '<?xml version="1.0" ?><sdf version="1.4"><model name="%s">' % model_name
        s += '<static>%s</static>' % ('true' if static else 'false')
        s += '<pose>0 0 0 0 0 0</pose>'
        s += '<link name="link"><collision name="collision"><geometry><box>'

        s += '<size>%.4f %.4f %.4f</size>' % (sx, sy, sz)
        s += '</box></geometry></collision><visual name="visual"><geometry><box>'
        s += '<size>%.4f %.4f %.4f</size>' % (sx, sy, sz)
        s += '</box></geometry>'
        s += '<material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>%s</name></material>' % material
        s += '</visual></link></model></sdf>'

        pose = Pose()
        pose.position.x = tx
        pose.position.y = ty
        pose.position.z = tz
        quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        pose.orientation.x = quaternion[0]
        pose.orientation.y = quaternion[1]
        pose.orientation.z = quaternion[2]
        pose.orientation.w = quaternion[3]

        self.spawn_model(model_name, s, pose)

    def spawn_model(self, model_name, model_xml, initial_pose):
        rospy.wait_for_service('gazebo/spawn_sdf_model')
        spawn_model = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
        spawn_model(model_name, model_xml, "quadrotor", initial_pose, "world")


def main(args):
    """ Main function. """
    # Initialize our ROS node.
    rospy.init_node('simulation_randomization', anonymous=True)

    randomize_simulation = SimulationRandomizer()
    randomize_simulation()


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
