#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Sean Kirmani <sean@kirmani.io>
#
# Distributed under terms of the MIT license.
"""
TODO(kirmani): DESCRIPTION GOES HERE

List of tests we are running:
- spawn cylinder
- spawn box
- spawn kitcnen counter elevation and have it go up
- spawn corner by wall w/ half FOV
- spawn in laundry room
- spawn under table, avoids legs?
- spawn free space in laundry room
- spawn go between couches
- spawn inside kitchen
"""
#chkpt path hardcoded to "-t -m $(find journey)/tensorflow/models/collision_avoidance/discrete/model.ckpt-999"
from simulation_randomization import SimulationRandomizer
#from train_obstacle_avoidance import DeepDronePlanner
import tensorflow as tf
from environment import Environment
from replay_buffer import ReplayBuffer
from ddpg import DeepDeterministicPolicyGradients
from geometry_msgs.msg import Pose


class Test:

    def __init__(self, name):
        self.name = name
        self.start = (1, 1, 1)
        self.goal = (2, 2, 2)


class ModelValidator:

    def __init__(self):
        corner_test = Test("corner")
        kitchen_nook_test = Test("kitchen_nook")

        self.tests = [corner_test, kitchen_nook_test]

        print("Validator Initialized")

    #maybe have to pass in the entire deep drone planner
    def validate(self, env, ddpg):
        #model already loaded

        #set goal, starting pose, and initialize drone using env.reset in ddpg.test
        test = 1
        start = (1.0, 1.0, 1.0)
        goal = (2.0, 6.0, 1.0)

        #run the test (code mostly from ddpg.eval)
        test_name = "Bathroom"
        num_success = ddpg.test(
            env=env,
            test_name=test_name,
            start=start,
            goal=goal,
            num_attempts=100,
            max_episode_len=1000)

        print("Test: %s num_successes: %d" % (test_name, num_success))
        #for test in self.tests:
        #    print("Running test: %s" % test.name)

    def laundry_room_test(self):
        self.nav_goal.position.x = -0.75
        self.nav_goal.position.y = 5.0
        self.nav_goal.position.z = 1.0

    def __call__(self):
        print("test call to self")
        self.laundry_room_test()


#how much stuff has already been done

# stuff I need to do:
# set goal
# set starting position
# run the thing ?? not so sure
# (later) set up environment
# check if step terminated and check reward

# know when drone has collided
# set goal:
# self.nav_goal.position.x = goal_position[0]
# self.nav_goal.position.y = goal_position[1]
# self.nav_goal.position.z = goal_position[2]

# set start position (from sim rand)
