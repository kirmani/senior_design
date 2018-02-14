import gym
import rospy
import roslaunch
import time
import numpy as np

from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import LaserScan

from gym.utils import seeding

class GazeboJourneyDiscreteEnv(gazebo_env.GazeboEnv):

    def __init__(self, distance_threshold=0.5, rate=4, discrete_controls=True):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboJourneyDiscrete_v0.launch")
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        self.action_space = spaces.Discrete(7) #F,L,R
        self.reward_range = (0, np.inf)

        self.distance_threshold = distance_threshold  # meters
        self.update_rate = rate  # Hz
        self.discrete_controls = discrete_controls

        # Set max linear velocity to 0.5 meters/sec.
        self.max_linear_velocity = 0.5
        rospy.set_param('control_vz_max', self.max_linear_velocity * 1000)
        print("Max linear velocity (mm/s): %s" %
              rospy.get_param('control_vz_max'))

        # Set max angular velocity to 30 degrees/sec.
        self.max_angular_velocity = np.pi / 6.0
        rospy.set_param('euler_angle_max', self.max_angular_velocity)
        print("Max angular velocity (mm/s): %s" %
              rospy.get_param('euler_angle_max'))

        # Initialize our ROS node.
        rospy.init_node('deep_drone_planner', anonymous=True)

        # Inputs.
        self.image_subscriber = rospy.Subscriber('/ardrone/front/image_raw',
                                                 Image, self.on_new_image)
        self.image_msg = None

        # Subscribe to ground truth pose.
        self.ground_truth_subscriber = rospy.Subscriber(
            '/ground_truth/state', Odometry, self.on_new_state)
        self.last_collision_pose = Pose()
        self.pose = None

        # Subscribe to collision detector.
        self.collision_subscriber = rospy.Subscriber(
            '/ardrone/crash_sensor', ContactsState, self.on_new_contact_data)
        self.collided = False

        # Actions.
        self.velocity_publisher = rospy.Publisher(
            '/cmd_vel', Twist, queue_size=10)

        # Reset topics.
        self.takeoff_publisher = rospy.Publisher(
            '/ardrone/takeoff', EmptyMessage, queue_size=10)

        self.land_publisher = rospy.Publisher(
            '/ardrone/land', EmptyMessage, queue_size=10)

        # Publish the collision state.
        self.collision_state_publisher = rospy.Publisher(
            '/ardrone/collision_state', CollisionState, queue_size=10)

        # The rate which we publish commands.
        self.rate = rospy.Rate(self.update_rate)

        # Simulation reset randomization.
        self.randomize_simulation = SimulationRandomizer()

        self._seed()

    def discretize_observation(self,data,new_ranges):
        # discretized_ranges = []
        # min_range = 0.2
        # done = False
        # mod = len(data.ranges)/new_ranges
        # for i, item in enumerate(data.ranges):
        #     if (i%mod==0):
        #         if data.ranges[i] == float ('Inf'):
        #             discretized_ranges.append(6)
        #         elif np.isnan(data.ranges[i]):
        #             discretized_ranges.append(0)
        #         else:
        #             discretized_ranges.append(int(data.ranges[i]))
        #     if (min_range > data.ranges[i] > 0):
        #         done = True
        return discretized_ranges,done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

    def _reset(self):

        self.velocity_publisher.publish(Twist())

        # Randomize simulation environment.
        self.randomize_simulation()

        # Clear our frame buffer.
        self.frame_buffer.clear()

        # Take-off.
        self.unpause_physics()
        self.takeoff_publisher.publish(EmptyMessage())

        # Get state.
        state = self.get_current_state()

        # Reset collision state.
        self.collided = False

        return state
