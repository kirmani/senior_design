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


class Test:
    def __init__(self, name):
        self.name = name
        self.start = 
        self.goal =


class ModelValidator:
    def __init__(self):
        corner_test = Test("corner")
        kitchen_nook_test = Test("kitchen_nook")

        self.tests = [corner_test, kitchen_nook_test]

        self.randomize_simulation = SimulationRandomizer()
        print("Validator Initialized")

    def validate(self, env):
        for test in self.tests:
            print("Running test: %s" % test.name)


    def laundry_room_test(self):
        self.nav_goal.position.x = -0.75 
        self.nav_goal.position.y = 5.0    
        self.nav_goal.position.z = 1.0


    def __call__(self):
        print("test call to self")
        self.laundry_room_test()

#how much stuff has already been done

# stuff I need to do:
# know when drone has collided
# set goal:
# self.nav_goal.position.x = goal_position[0]
# self.nav_goal.position.y = goal_position[1]
# self.nav_goal.position.z = goal_position[2]

# set start position (from sim rand)
