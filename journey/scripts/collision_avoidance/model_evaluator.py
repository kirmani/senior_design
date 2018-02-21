#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Sean Kirmani <sean@kirmani.io>
#
# Distributed under terms of the MIT license.
"""
TODO(kirmani): DESCRIPTION GOES HERE
"""

class Test:
    def __init__(self, name):
        self.name = name

class ModelValidator:
    def __init__(self):
        corner_test = Test("corner")
        kitchen_nook_test = Test("kitchen_nook")

        self.tests = [corner_test, kitchen_nook_test]
        print("Validator Initialized")

    def validate(self):
        for test in self.tests:
            print("Running test: %s" % test.name)
