#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Sean Kirmani <sean@kirmani.io>
#
# Distributed under terms of the MIT license.
class Environment:

    def __init__(self, reset, step, reward, terminal, visualize=None):
        self.reset = reset
        self.step = step
        self.reward = reward
        self.terminal = terminal
        self.visualize = visualize

    def reset(self):
        return self.reset()

    def step(self, state, action):
        return self.step(state, action)

    def reward(self, state, action):
        return self.reward(state, action)

    def terminal(self, state, action):
        return self.terminal(state, action)

    def visualize(self, state, action):
        if self.visualize:
            self.visualize(state, action)
