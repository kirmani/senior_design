#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Sean Kirmani <sean@kirmani.io>
#
# Distributed under terms of the MIT license.
"""
Navigation planner.
"""

import rospy


class NavigationPlannerNode:

    def __init__(self):
        # Initialize ROS node.
        rospy.init_node('nav_planner', anonymous=True)

        print("Navigation planner initialzated.")


def main():
    nav_planner = NavigationPlannerNode()


if __name__ == '__main__':
    main()
