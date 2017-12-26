#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Sean Kirmani <sean@kirmani.io>
#
# Distributed under terms of the MIT license.
"""
TODO(kirmani): DESCRIPTION GOES HERE
"""
import argparse
import rospy
import sys
import traceback
import time
import tf
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose


class SimulationRandomizer:

    def __init__(self):
        print("Initialized simulation randomizer.")

    def __call__(self):
        print("Randomized simulation.")
        hallway_width = 2.0
        wall_length = 100.0
        wall_width = 0.1
        wall_height = 3.0

        # Create left wall.
        self.spawn_box(
            model_name='left_wall',
            ty=(-(hallway_width + wall_width) / 2),
            tz=(wall_height / 2),
            sx=wall_length,
            sy=wall_width,
            sz=wall_height,
            material='Gazebo/Blue')

        # Create right wall.
        self.spawn_box(
            model_name='right_wall',
            ty=((hallway_width + wall_width) / 2),
            tz=(wall_height / 2),
            sx=wall_length,
            sy=wall_width,
            sz=wall_height,
            material='Gazebo/Red')

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
