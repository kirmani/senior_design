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
from std_srvs.srv import Empty

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
        self.num_boxes = 0

        # Publish model state.
        self.model_state_publisher = rospy.Publisher(
            '/gazebo/set_model_state', ModelState, queue_size=10)

        # Delete models.
        rospy.wait_for_service('gazebo/delete_model')
        self.delete_model = rospy.ServiceProxy('gazebo/delete_model',
                                               DeleteModel)

        # Pause physics.
        rospy.wait_for_service('gazebo/pause_physics')
        self.pause_physics = rospy.ServiceProxy('gazebo/pause_physics', Empty)

        # Unpause physics.
        rospy.wait_for_service('gazebo/unpause_physics')
        self.unpause_physics = rospy.ServiceProxy('gazebo/unpause_physics',
                                                  Empty)

        print("Initialized simulation randomizer.")

    def __call__(self):
        print("Randomized simulation.")

        self.pause_physics()

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

        floorplan_2d, rows, cols = self.generate_floorplan()

        # Create a 3D world that corresponds to our 2D floorplan.
        # TODO(armandb): Display a couple of big boxes instead of many small
        #                boxes as an optimization.

        # Delete existing objects.
        self.delete_model('floor')
        self.delete_model('bottom_border')
        self.delete_model('top_border')
        self.delete_model('left_border')
        self.delete_model('right_border')
        self.delete_model('ceiling')
        for i in range(self.num_boxes):
            string = 'box%d' % i
            self.delete_model(string)
        self.num_boxes = 0

        # Transform our floorplan into 3D boxes.
        # Left corner starts at (0, 0).
        for row in range(1, rows - 1):
            for col in range(1, cols - 1):
                if (floorplan_2d[row, col] == 0):
                    self.spawn_box(
                        model_name='box%d' % (self.num_boxes),
                        tx=hallway_width * col + hallway_width / 2,
                        ty=hallway_width * row + hallway_width / 2,
                        tz=wall_height / 2,
                        sx=hallway_width,
                        sy=hallway_width,
                        sz=wall_height,
                        material=random.choice(MATERIALS))
                    self.num_boxes += 1

        # Bottom border.
        self.spawn_box(
            model_name='bottom_border',
            tx=hallway_width * cols / 2,
            ty=hallway_width / 2,
            tz=wall_height / 2,
            sx=hallway_width * cols,
            sy=hallway_width,
            sz=wall_height,
            material=random.choice(MATERIALS))

        # Top border.
        self.spawn_box(
            model_name='top_border',
            tx=hallway_width * cols / 2,
            ty=hallway_width * rows - hallway_width / 2,
            tz=wall_height / 2,
            sx=hallway_width * cols,
            sy=hallway_width,
            sz=wall_height,
            material=random.choice(MATERIALS))

        # Left border.
        self.spawn_box(
            model_name='left_border',
            tx=hallway_width / 2,
            ty=hallway_width * rows / 2,
            tz=wall_height / 2,
            sx=hallway_width,
            sy=hallway_width * (rows - 2),
            sz=wall_height,
            material=random.choice(MATERIALS))

        # Right border.
        self.spawn_box(
            model_name='right_border',
            tx=hallway_width * cols - hallway_width / 2,
            ty=hallway_width * rows / 2,
            tz=wall_height / 2,
            sx=hallway_width,
            sy=hallway_width * (rows - 2),
            sz=wall_height,
            material=random.choice(MATERIALS))

        # Floor.
        self.spawn_box(
            model_name='floor',
            tx=hallway_width * float(cols) / 2,
            ty=hallway_width * float(rows) / 2,
            tz=-0.1,
            sx=hallway_width * cols,
            sy=hallway_width * rows,
            sz=0.2,
            material=random.choice(MATERIALS))

        # Ceiling.
        self.spawn_box(
            model_name='ceiling',
            tx=hallway_width * float(cols) / 2,
            ty=hallway_width * float(rows) / 2,
            tz=0.1 + wall_height,
            sx=hallway_width * cols,
            sy=hallway_width * rows,
            sz=0.2,
            material=random.choice(MATERIALS))

        # Choose safe location for quadrotor to spawn.
        drone_row = np.random.randint(1, rows)
        drone_col = np.random.randint(1, cols)
        while (floorplan_2d[drone_row, drone_col] != 1):
            drone_row = np.random.randint(1, rows)
            drone_col = np.random.randint(1, cols)

        # Min amount of room on each side.
        slack = 0.5

        # Where offset is from bottom left corner.
        drone_offsetrow = slack + self.quadrotor_width / 2 + np.random.random(
        ) * ((hallway_width - slack * 2 - self.quadrotor_width) - 0) + 0
        drone_offsetcol = slack + self.quadrotor_width / 2 + np.random.random(
        ) * ((hallway_width - slack * 2 - self.quadrotor_width) - 0) + 0

        # Spawn our quadrotor.
        self.spawn_quadrotor(
            tx=hallway_width * drone_col + drone_offsetcol,
            ty=hallway_width * drone_row + drone_offsetrow,
            tz=1.0,
            yaw=quadrotor_yaw)

        # Unpause physics.
        self.unpause_physics()

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

        # Visualize the generated floorplan.
        print(floorplan)

        return floorplan, rows, cols

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
