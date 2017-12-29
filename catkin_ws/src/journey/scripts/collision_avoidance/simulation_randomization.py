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
        self.num_boxes = 0
        # Publish model state.
        self.model_state_publisher = rospy.Publisher(
            '/gazebo/set_model_state', ModelState, queue_size=10)

        print("Initialized simulation randomizer.")

    def __call__(self):
        print("Randomized simulation.")
        twoD_floorplan = self.generate_floorplan() #TODO for testing only
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

        #randomized params for pillars
#        pillar_height = wall_height
#        if (hallway_width - self.quadrotor_width * 1.5) / 2 > 1:
#                pillar_max_rad = 1
#        else:
#                pillar_max_rad = hallway_width - self.quadrotor_width * 1.5) / 2
        
        #turn 2d floorplan into 3D
        #TODO later optimize the display of multiple boxes in a row as 1 big box

        #delete all objects already there
        rospy.wait_for_service('gazebo/delete_model')
        delete_model = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)

        for i in range(0,self.num_boxes):
                string = 'box%d' %(i)
                delete_model(string)
        self.num_boxes = 0

        #turn new floorplan into 3D
        for row in range(0,10):
                for col in range(0,18): #TODO need to change these bounds to rows and cols that were used in randomizer 
                        if(twoD_floorplan[row,col] == 2):
                                self.spawn_box(model_name='box%d' % (self.num_boxes),
                                               tx= hallway_width*row,
                                               ty= hallway_width*col,
                                               tz= wall_height/2,
                                               sx= hallway_width,
                                               sy= hallway_width,
                                               sz= wall_height, 
                                               material=random.choice(MATERIALS))
                                self.num_boxes += 1
        exit() #TODO remove this exit


        # Delete models.
#        rospy.wait_for_service('gazebo/delete_model')
#        delete_model = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
#        delete_model('left_wall')
#        delete_model('right_wall')
#        delete_model('floor')
#        delete_model('ceiling')
#        delete_model('deadend_front')
#        delete_model('deadend_back')
#        delete_model('pillar')

        # Create left wall.
#        self.spawn_box(
#            model_name='left_wall',
#            ty=(-(hallway_width + wall_width) / 2),
#            tz=(wall_height / 2),
#            sx=wall_length,
#            sy=wall_width,
#            sz=wall_height,
#            material=random.choice(MATERIALS))

        # Create right wall.
#        self.spawn_box(
#            model_name='right_wall',
#            ty=((hallway_width + wall_width) / 2),
#            tz=(wall_height / 2),
#            sx=wall_length,
#            sy=wall_width,
#            sz=wall_height,
#            material=random.choice(MATERIALS))

        # Create floor.
#        self.spawn_box(
#            model_name='floor',
#            tz=(-wall_width / 2),
#            sx=wall_length,
#            sy=hallway_width,
#            sz=wall_width,
#            material=random.choice(MATERIALS))

        # Create ceiling.
#        self.spawn_box(
#            model_name='ceiling',
#            tz=(wall_height + wall_width / 2),
#            sx=wall_length,
#            sy=hallway_width,
#            sz=wall_width,
#            material=random.choice(MATERIALS))

        # Create dead-end (front).
#        self.spawn_box(
#            model_name='deadend_front',
#            tx=(wall_length / 2),
#            tz=(wall_height / 2),
#            sx=wall_width,
#            sy=hallway_width,
#            sz=wall_height,
#            material=random.choice(MATERIALS))

        # Create dead-end (back).
#        self.spawn_box(
#            model_name='deadend_back',
#            tx=(-wall_length / 2),
#            tz=(wall_height / 2),
#            sx=wall_width,
#            sy=hallway_width,
#            sz=wall_height,
#            material=random.choice(MATERIALS))

        # Create Cylinder close to end of the room
#        self.spawn_cylinder(
#            model_name='pillar',
#            tx=((-wall_length / 2)+.5),
#            tz=(wall_height / 2),
#            radius=.1, #use something fixed for now
#            length=wall_height,
#            material=random.choice(MATERIALS))

        self.spawn_quadrotor(ty=quadrotor_ty, yaw=quadrotor_yaw)

        # Wait a little bit for environment to stabilize.
        rospy.sleep(2.)


    def generate_floorplan(self, rows=10, cols=18):
        print("Generate floorplan.")


        # TODO(armand): Generate floorplan here. Set grid cells that we
        # consider hallways to 1.
        
        ##algorithm (I want the grid to map to real space) TODO: see if generating 10 small boxes is easier than generating 1 big box
                #maybe 3 way or 4 way hallways are helpful
                #can end up with hanging hallways, (have to fix that)
                #hallways should be able to connect to one another (not just from the edge) like a square
        #Step1
        #have a chunk, split it into 2 chunks with a horizontal hallway
        #split each of those chunks into 2 with a vertical hallway
        #split each of those chunks into 2 with a horizontal hallway
        #...

        #have a bunch of 0's
        floorplan = np.zeros((rows, cols))
        
        #get a number which tells me how many horizontal and veritical splits I want to do (num_hallways) - for now, let's just say 6 or 8 hallways total, or randomly decides
        num_hallways = 6 #for testing, can be variable

        #set the border all to 2's, so we know when to stop on our hallway creation 
        for x in range(0, cols-1):
                floorplan[0,x] = 2
                floorplan[rows-1,x] = 2

        for x in range(0, rows):
                floorplan[x,0] = 2
                floorplan[x, cols-1] = 2
        #print(floorplan)
        #exit()
        row_or_col = np.random.randint(0,1) #define row as 0, col as 1, pick direction of 1st hallway
        ROW = 0
        COL = 1        
        for i in range(0, num_hallways):
                if(row_or_col == 0): #row                

                        direction = np.random.randint(0,2) #up left = -1, down right = 1
                        if(direction == 0):
                                direction = -1
                                x = cols-2
                        else:
                                x = 1

                        #TODO take into account variable width of the walls
                        #corner case when it can't find a suitable row and does endless loop - added the calls var
                        #MATH CHECKS OUT corner case where hallway it runs into is on the last possible row - changed code to have x += direction at end of loop if floorplan(x,y) == 1
                        #TODO fix hanging hallways
                        #TODO if hallway is on the edge rows, it breaks your hallway sensing scheme to decide if it's ok to put a hallway, look 1 square deeper
                        rand_num = np.random.randint(1,rows-1) #which row to create hallway
                        while(floorplan[rand_num,x] != 2):
                                rand_num = np.random.randint(1,rows-1) #need to do it again b/c code not super clean
                                calls = 0
                                while not ( floorplan[rand_num-1, x] != 1 and floorplan[rand_num, x] != 1 and floorplan[rand_num+1, x] != 1):
                                        rand_num = np.random.randint(1,rows-1) #it's for which row to start on
                                        calls += 1
                                        if(calls == 10):
                                                break
                                if(calls == 10):
                                        print('calls rows == 10')
                                        break
                                while(floorplan[rand_num,x] == 0): #while we're not at the end of array or at another hallway                                
                                        floorplan[rand_num,x] += 1
                                        x += direction
                                if(floorplan[rand_num,x] == 1):
                                        x += direction
                        row_or_col = COL

                else: #do vertical hallway
                        #using a random number between 0 and rows (or cols) and one for direction
                        #need to write logic for what to do to make sure you don't go over another hallway 

                        direction = np.random.randint(0,1) #up left = -1, down right = 1
                        if(direction == 0):
                                direction = -1
                                x = rows-2
                        else:
                                x = 1

                        #TODO take into account variable width of the walls
                        # corner case when it can't find a suitable row - added the calls var
                        #MATH CHECKS OUT corner case where hallway it runs into is on the last possible row - changed code to have x += direction at end of loop if floorplan(x,y) == 1
                        #TODO fix hanging hallways
                        #TODO if hallway is on the edge rows, it breaks your hallway sensing scheme to decide if it's ok to put a hallway, look 1 square deeper
                        rand_num = np.random.randint(1,cols-1) #which row to create hallway
                        while(floorplan[x,rand_num] != 2):
                                rand_num = np.random.randint(1,cols-1) #need to do it again b/c code not super clean
                                while not ( floorplan[x,rand_num-1] != 1 and floorplan[x,rand_num] != 1 and floorplan[x,rand_num+1] != 1):
                                        rand_num = np.random.randint(1,cols-1) #it's for which row to start on
                                        calls += 1
                                        if(calls == 10):
                                                break
                                if(calls == 10):
                                        print('calls cols == 10')
                                        break
                                while(floorplan[x,rand_num] == 0): #while we're not at the end of array or at another hallway                                
                                        floorplan[x,rand_num] += 1
                                        x += direction
                                if(floorplan[x,rand_num] == 1):
                                        x += direction
                        row_or_col = ROW

        #add walls on the outside of the hallways
        print(floorplan)
        print(' ')
        print(' ')
        for row in range(1,rows-1):
                for col in range(1,cols-1):
                        if(floorplan[row,col] == 0):
                                if(floorplan[row+1,col] == 1 or floorplan[row-1,col] == 1 or floorplan[row,col+1] == 1 or floorplan[row,col-1] == 1): #if adjacent to a 1
                                        floorplan[row,col] = 2


        # Visualize the generated floorplan.
        print(floorplan)

        return floorplan



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
