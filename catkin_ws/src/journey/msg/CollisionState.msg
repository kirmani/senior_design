# This message contains the optimal actions to execute to fly straight and
# avoid obstacles as learned by the network.

# The length of the time window we make predictions about the future.
uint32 horizon

# The probability that we will collide over our horizon.
float32 collision_probability

# The dimensionality of our action space.
uint32 action_dimensionality

# The optimal action to execute.
float32[] action
