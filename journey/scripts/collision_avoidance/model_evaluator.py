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

    def __init__(self, name, (start_x, start_y, start_z), (goal_x, goal_y,
                                                           goal_z)):
        self.name = name
        self.start = (start_x, start_y, start_z)
        self.goal = (goal_x, goal_y, goal_z)


class ModelValidator:

    def __init__(self):
        #Initialize test cases with name start and goal
        self.tests = []
        self.tests.append(Test("Across Living Room", (1, 1, 1), (4, 4, 1)))
        self.tests.append(Test("Through Kitchen", (1, 1, 1), (2, 6, 1)))
        self.tests.append(Test("Exit Laundry Room", (-.75, 5, 1), (1, 1, 1)))
        self.tests.append(Test("Under Table", (5.25, 7, 1), (5, 4, 1)))
        self.tests.append(Test("Around Corner", (5.5, 4, 1), (4, 2, 1)))
        self.tests.append(Test("Between Couches", (2, 1.5, 1), (2, 4.5, 1)))

        print("Validator Initialized")

    def validate(self, env, ddpg):
        num_test_attempts = 1
        total_success = 0
        total_attempts = 0
        for test in self.tests:
            #run the test (code mostly from ddpg.eval)
            num_success = ddpg.test(
                env=env,
                test_name=test.name,
                start=test.start,
                goal=test.goal,
                num_attempts=num_test_attempts,
                max_episode_len=1000)
            total_success += num_success
            total_attempts += num_test_attempts
            print("Test: %s num_successes: %d" % (test.name, num_success))
        if total_attempts != 0:
            print("Validation Complete. Score: %d%%" %
                  (total_success * 100 / total_attempts))


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
