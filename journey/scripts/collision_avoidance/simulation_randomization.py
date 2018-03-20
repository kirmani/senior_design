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
import matplotlib.pyplot as plt
import numpy as np
import rospy
import random
from scipy import ndimage
import sys
import traceback
import time
import tf
from gazebo_msgs.srv import DeleteModel
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose
from std_msgs.msg import Empty as EmptyMessage
from std_srvs.srv import Empty as EmptyService

from gazebo_msgs.srv import SetLightProperties
from std_msgs.msg import ColorRGBA

MATERIALS = [
    'Gazebo/WoodFloor',
    'Gazebo/CeilingTiled',
    'Gazebo/WoodPallet',
]

SPAWN_REGIONS = [
    'living_room',
    'laundry_room',
    'kitchen',
    'dining_room',
    'entry_way'
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
        self.max_quadrotor_start_yaw = 180  # degrees
        self.num_boxes = 0

        # Publish model state.
        self.model_state_publisher = rospy.Publisher(
            '/gazebo/set_model_state', ModelState, queue_size=10)

        self.randomizer_publisher = rospy.Publisher(
            '/journey/randomize', EmptyMessage, queue_size=10)

        print("Initialized simulation randomizer.")

    def delete_model(self, model):
        # Delete models.
        rospy.wait_for_service('gazebo/delete_model')
        delete_model = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
        delete_model(model)

    def pause_physics(self):
        # Pause physics.
        rospy.wait_for_service('gazebo/pause_physics')
        pause_physics = rospy.ServiceProxy('gazebo/pause_physics', EmptyService)
        pause_physics()

    def unpause_physics(self):
        # Unpause physics.
        rospy.wait_for_service('gazebo/unpause_physics')
        unpause_physics = rospy.ServiceProxy('gazebo/unpause_physics',
                                             EmptyService)
        unpause_physics()

    def GetRandomAptPosition(self):
        # Give each sample a PMF that corresponds to its area as
        # opposed to uniformly sampling, so doesn't bias to explore small
        # regions more often.
        room = np.random.random()

        if room < .65:
            spawn_room = 'living_room'
            min_x = 0.5
            max_x = 4.0
            min_y = 0.5
            max_y = 4.2
            min_z = 1.2 
            max_z = 2.5

        elif room < .87:
            spawn_room = 'kitchen'
            min_x = 0.5
            max_x = 4.0
            min_y = 6.0
            max_y = 7.2
            min_z = 0.5 
            max_z = 2.5

        elif room < .96: 
            spawn_room = 'laundry_room' 
            min_x = -0.8 
            max_x = -0.6
            min_y = 4.3 
            max_y = 4.8 
            min_z = 0.5
            max_z = 2.5

        elif room < .98:
            spawn_room = 'dining_room'
            min_x = 5.0
            max_x = 5.4
            min_y = 4.0
            max_y = 8.2
            min_z = 1.5
            max_z = 2.5

        else:
            spawn_room = 'entry_way'
            min_x = 3.2
            max_x = 4.5
            min_y = 8.1
            max_y = 8.3
            min_z = .5
            max_z = 2.5

        tx = min_x + (np.random.random() * (max_x - min_x))
        ty = min_y + (np.random.random() * (max_y - min_y))
        tz = min_z + (np.random.random() * (max_z - min_z))
        #NOTE: can't get drone to spawn facing the correct way
        #but doesn't matter b/c have goal now
        yaw = (2.0 * np.random.random() * self.max_quadrotor_start_yaw -
               self.max_quadrotor_start_yaw) * np.pi / 180.0

        return (tx, ty, tz, yaw)

    def __call__(self, start_x, start_y, start_z, test = 0):
        print("Randomized simulation.")

        self.pause_physics()

        self.randomizer_publisher.publish(EmptyMessage())

        if test == 0:
            self.set_intensity()

            # Pick randomized parameters.
            (quadrotor_tx, quadrotor_ty, quadrotor_tz,
            quadrotor_yaw) = self.GetRandomAptPosition()
        else:
            quadrotor_tx = start_x 
            quadrotor_ty = start_y
            quadrotor_tz = start_z
            quadrotor_yaw = 0

        # Spawn our quadrotor.
        self.spawn_quadrotor(
            tx=quadrotor_tx,
            ty=quadrotor_ty,
            tz=quadrotor_tz,
            yaw=quadrotor_yaw)
        # Unpause physics.
        self.unpause_physics()
        print("unpaused physics")
        # Wait a little bit for the drone spawn to stabilize. Maybe there's a
        # way to do this without sleeping?
        rospy.sleep(2)

    #might not work because lights might not be considered models
    def set_intensity(self):
        intensity = 100 + (np.random.random() * (235 - 100))
        diffuse = ColorRGBA()
        diffuse.r = intensity #all the same so greyscale
        diffuse.g = intensity
        diffuse.b = intensity
        diffuse.a = 255 #transparency 0 is completely transparent
        #changing attenuation doesn't seem do do anything
        atten_const = 0.9
        atten_lin = 0.01
        atten_quad = 0.0

        rospy.wait_for_service('gazebo/set_light_properties')
        set_light_properties = rospy.ServiceProxy('gazebo/set_light_properties',
                                                  SetLightProperties)
        success = set_light_properties('sun', diffuse, atten_const, atten_lin,
                                       atten_quad)
        if success:
            print("set light properties was a success!")

    def spawn_quadrotor(self, tx=0, ty=0, tz=1, roll=0, pitch=0, yaw=0):
        position = (tx, ty, tz)
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

    def spawn_cylinder(self,
                       model_name="cylinder",
                       tx=0,
                       ty=0,
                       tz=0.5,
                       radius=.1,
                       length=2,
                       yaw=0,
                       pitch=0,
                       roll=0,
                       static=True,
                       material='Gazebo/Blue'):
        s = '<?xml version="1.0" ?><sdf version="1.4"><model name="%s">' % model_name
        s += '<static>%s</static>' % ('true' if static else 'false')
        s += '<link name="pillar">'
        s += '<pose>0 0 0 0 0 0</pose>'

        s += '<collision name="collision"><geometry><cylinder>'
        s += '<radius>%.4f</radius>' % (radius)
        s += '<length>%.4f</length>' % (length)
        s += '</cylinder></geometry></collision>'

        s += '<visual name="visual"><geometry><cylinder>'
        s += '<radius>%.4f</radius>' % (radius)
        s += '<length>%.4f</length>' % (length)
        s += '</cylinder></geometry>'

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

    def generate_floorplan(self, rows=10, cols=18, num_hallways=6):
        """
        Algorithm:
        For horizontal hallways (same idea for vertical)
        1. Start at one end of the grid, go straight until you hit the other
           end of the grid or a vertical hallway.
        2. If you hit another hallway, change your row and keep going in the
           same direction.
        3. Do 1 and 2 until you hit the end of the grid.
        4. Now do same with vertical.
        5. Handle corner cases.

        It was relatively simple and created more randomization than just
        starting from a wall and drawing a line until you hit something and
        repeat.

        num_hallways tells me how many horizontal and veritical splits I want
        to do - for now, let's just say 6 or 8 hallways total, or randomly
        decides.
        """
        print("Generate floorplan.")
        rows += 2
        cols += 2
        floorplan = np.zeros((rows, cols))

        # Set the border all to 2's, so we know when to stop on our hallway
        # creation.
        for x in range(0, cols - 1):
            floorplan[0, x] = 2
            floorplan[rows - 1, x] = 2

        for x in range(0, rows):
            floorplan[x, 0] = 2
            floorplan[x, cols - 1] = 2

        # Define row as 0, col as 1, pick direction of 1st hallway.
        row_or_col = np.random.randint(0, 2)
        ROW = 0
        COL = 1
        for i in range(0, num_hallways):
            if (row_or_col == 0):
                # Row case.
                # Up left = -1, Down right = 1.
                direction = np.random.randint(0, 2)
                if (direction == 0):
                    direction = -1
                    x = cols - 2
                else:
                    x = 1

                # TODO(armand): Try to make hallways different sizes from one
                #               another.

                # corner case when it can't find a suitable row and does
                # endless loop - added the calls var

                # MATH CHECKS OUT corner case where hallway it runs into is on
                # the last possible row - changed code to have x += direction
                # at end of loop if floorplan(x,y) == 1
                # probs works, need extensive test - fix hanging hallways

                # TODO(armand): if hallway is on the edge rows, it breaks your
                #               hallway sensing scheme to decide if it's ok to
                #               put a hallway, look 1 square deeper
                rand_num = np.random.randint(1, rows - 1)
                while (floorplan[rand_num, x] != 2):

                    # Find valid row to create hallway, if can't find valid
                    # row, stop trying after 10 to avoid infinite loop.

                    # TODO(armand): Need to do it again because code not super
                    #               clean.
                    rand_num = np.random.randint(1, rows - 1)
                    calls = 0
                    while not (floorplan[rand_num - 1, x] != 1 and
                               floorplan[rand_num, x] != 1 and
                               floorplan[rand_num + 1, x] != 1):
                        # It's for which row to start on.
                        rand_num = np.random.randint(1, rows - 1)
                        calls += 1

                        # Could chnage so it doesn't call 10x but instead gets
                        # 10 numbers and tries them all.
                        if (calls == 10):
                            break
                    if (calls == 10):
                        print('calls rows == 10')
                        break

                    # Hanging hallway corner case, if hallway is hanging,
                    # connect it to the end or another hallway behind it.
                    behind = -direction
                    while (floorplan[rand_num, x + behind] == 0):
                        floorplan[rand_num, x + behind] = 1
                        behind -= direction

                    # Create the hallway.
                    while (floorplan[rand_num, x] == 0):
                        # While we're not at the end of array or at another
                        # hallway.
                        floorplan[rand_num, x] += 1
                        x += direction
                    if (floorplan[rand_num, x] == 1):
                        x += direction
                row_or_col = COL

            else:
                # Do vertical hallway.

                # Using a random number between 0 and rows (or cols) and one
                # for direction.

                # Need to write logic for what to do to make sure you don't go
                # over another hallway.

                # Up left = -1, down right = 1.
                direction = np.random.randint(0, 1)
                if (direction == 0):
                    direction = -1
                    x = rows - 2
                else:
                    x = 1

                # TODO(armand): try to make hallways different sizes from one
                #               another

                # corner case when it can't find a suitable row - added the
                # calls var

                # MATH CHECKS OUT corner case where hallway it runs into is on
                # the last possible row - changed code to have x += direction
                # at end of loop if floorplan(x,y) == 1

                # probs works, need extensive test - fix hanging hallways
                # TODO(armand): if hallway is on the edge rows, it breaks your
                # hallway sensing scheme to decide if it's ok to put a hallway,
                # look 1 square deeper

                # Which row to create hallway.
                rand_num = np.random.randint(1, cols - 1)
                while (floorplan[x, rand_num] != 2):
                    # Need to do it again b/c code not super clean.
                    rand_num = np.random.randint(1, cols - 1)
                    calls = 0
                    while not (floorplan[x, rand_num - 1] != 1 and
                               floorplan[x, rand_num] != 1 and
                               floorplan[x, rand_num + 1] != 1):
                        # It's for which row to start on.
                        rand_num = np.random.randint(1, cols - 1)
                        calls += 1
                        if (calls == 10):
                            break
                    if (calls == 10):
                        print('calls cols == 10')
                        break

                    # Hanging hallway corner case, if hallway is hanging,
                    # connect it to the end or another hallway behind it.
                    behind = -direction
                    while (floorplan[x + behind, rand_num] == 0):
                        floorplan[x + behind, rand_num] = 1
                        behind -= direction

                    # Create the hallway.
                    while (floorplan[x, rand_num] == 0):
                        # While we're not at the end of array or at another
                        # hallway.
                        floorplan[x, rand_num] += 1
                        x += direction
                    if (floorplan[x, rand_num] == 1):
                        x += direction
                row_or_col = ROW

        # Add walls on the outside of the hallways.
        for x in range(cols - 1):
            floorplan[0, x] = 0
            floorplan[rows - 1, x] = 0

        # Set walls to 1 and hallways to 0. Shape our floor plan back to our
        # desired grid size.
        floorplan[floorplan > 0] = 1
        floorplan = 1 - floorplan[1:-1, 1:-1]

        return floorplan


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
