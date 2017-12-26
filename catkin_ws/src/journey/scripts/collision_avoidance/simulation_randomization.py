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
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose


class SimulationRandomizer:

    def __init__(self):
        print("Initialized simulation randomizer.")

    def spawn_model(self, model_name, model_xml, initial_pose):
        rospy.wait_for_service('gazebo/spawn_sdf_model')
        spawn_model = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
        spawn_model(model_name, model_xml, "quadrotor", initial_pose, "world")

    def __call__(self):
        print("Randomized simulation.")
        hallway_width = 1.0
        wall_width = 0.1
        wall_height = 4.0
        left_wall_xml = self.create_box(
            model_name='left_wall',
            ty=-(hallway_width + wall_width) / 2,
            tz=wall_height / 2,
            sy=wall_width,
            sz=wall_height,
            material='Gazebo/Blue')
        self.spawn_model('left_wall', left_wall_xml, Pose())

        right_wall_xml = self.create_box(
            model_name='right_wall',
            ty=(hallway_width + wall_width) / 2,
            tz=wall_height / 2,
            sy=wall_width,
            sz=wall_height,
            material='Gazebo/Red')
        self.spawn_model('right_wall', right_wall_xml, Pose())

    def create_box(self,
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
        s += '<pose> %.4f %.4f %.4f %.4f %.4f %.4f</pose>' % (tx, ty, tz, yaw,
                                                              pitch, roll)
        s += '<link name="link"><collision name="collision"><geometry><box>'

        s += '<size>%.4f %.4f %.4f</size>' % (sx, sy, sz)
        s += '</box></geometry></collision><visual name="visual"><geometry><box>'
        s += '<size>%.4f %.4f %.4f</size>' % (sx, sy, sz)
        s += '</box></geometry>'
        s += '<material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>%s</name></material>' % material
        s += '</visual></link></model></sdf>'
        return s


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
