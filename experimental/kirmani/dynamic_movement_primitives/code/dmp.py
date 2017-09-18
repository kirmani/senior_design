#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Sean Kirmani <sean@kirmani.io>
#
# Distributed under terms of the MIT license.
"""
TODO(kirmani): DESCRIPTION GOES HERE
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf
import traceback
import time


class DynamicMovementPrimitive2D:

    def __init__(self,
                 K,
                 name,
                 alpha=5,
                 beta=0.5,
                 gamma=0.2,
                 verbose=False,
                 debug=False):
        self.K = K
        self.D = 2 * np.sqrt(K)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.name = name
        self.verbose = verbose
        self.debug = debug
        self.saved_f = None
        self.saved_s = None
        self.training_iterations = 0

        # Start Tensorflow session.
        self.sess = tf.Session()

        # Create Tensorflow model.
        model_params = {
            "use_neural_net": True,
            "learning_rate": 0.1,
            "momentum": 0.9,
            "num_basis_functions": 48,
        }
        self.estimator = tf.estimator.Estimator(
            model_fn=self.model_fn, params=model_params)

    def model_fn(self, features, labels, mode, params):
        """Model function for estimator."""

        if params["use_neural_net"]:
            first_hidden_layer = tf.layers.dense(
                features["x"], 48, activation=tf.nn.relu)
            second_hidden_layer = tf.layers.dense(
                first_hidden_layer, 48, activation=tf.nn.relu)
            third_hidden_layer = tf.layers.dense(
                second_hidden_layer, 48, activation=tf.nn.relu)
            predictions = tf.layers.dense(third_hidden_layer, 2)
        else:
            widths = tf.constant([[1.0, 1.0, 1.0]], tf.float64)
            centers = tf.constant([[0.0, 0.5, 1.0]], tf.float64)
            weights = tf.Variable(
                tf.cast(tf.random_normal(shape=[1, 3]), tf.float64),
                dtype=tf.float64)

            rbf = tf.matmul(features["x"],
                            tf.constant([[1.0, 1.0, 1.0]], tf.float64))
            rbf = tf.subtract(rbf, centers)
            rbf = tf.multiply(rbf, rbf)
            rbf = tf.multiply(tf.negative(widths), rbf)
            rbf = tf.exp(rbf)
            predictions = (tf.reduce_sum(
                tf.multiply(rbf, weights) * features["x"], 1, keep_dims=True) /
                           tf.reduce_sum(rbf, 1, keep_dims=True))
            print(predictions)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode, predictions={"Fs": predictions})

        loss = tf.losses.mean_squared_error(labels, predictions)

        eval_metric_ops = {
            "rmse":
            tf.metrics.root_mean_squared_error(
                tf.cast(labels, tf.float64), predictions)
        }

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=params["learning_rate"], momentum=params["momentum"])
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops)

    def TrainOnce(self, x, t):
        if self.verbose:
            print("Training DMP...")
        x0 = x[0]
        g = x[-1]
        tau = t[-1] - t[0]
        s = np.exp(-self.alpha * t / tau)

        # Compute velocity.
        x_dot = np.zeros((len(t), 2))
        for i in range(len(t)):
            x_dot[i] = (x[i] - x[i - 1]) / (t[i] - t[i - 1])

        # Compute acceleration.
        x_dot_dot = np.zeros((len(t), 2))
        for i in range(len(t)):
            x_dot_dot[i] = (x_dot[i] - x_dot[i - 1]) / (t[i] - t[i - 1])

        v = x_dot * tau
        v_dot = x_dot_dot * tau

        f_target = (tau * v_dot + self.D * v) / self.K - (g - x) + np.outer(
            s, g - x0)
        self.training_iterations += 1

        if self.training_iterations == 1:
            self.saved_f = f_target
            self.saved_s = s

        # Train tensorflow model.
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': np.array([[s_point] for s_point in s])},
            y=f_target,
            num_epochs=None,
            shuffle=True)
        self.estimator.train(input_fn=train_input_fn, steps=2000)

        if self.verbose:
            print("Finished training DMP!")
            print("Starting evaluation...")
            test_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={'x': np.array([[x] for x in s])},
                y=f_target,
                num_epochs=1,
                shuffle=False)
            ev = self.estimator.evaluate(input_fn=test_input_fn)
            print("Loss: %s" % ev["loss"])
            print("Root mean squared error: %s" % ev["rmse"])
            print("Finished evaluation")

    def _F(self, s):
        if self.training_iterations == 0:
            return np.zeros(len(s))
        elif self.training_iterations == 1:
            # Linearly interpolate.
            Fs = np.zeros((len(s), 2))

            for j in range(len(Fs)):
                value_set = False
                i = 0
                while i < len(self.saved_s) and not value_set:
                    if self.saved_s[i] < s[j]:
                        p = ((s[j] - self.saved_s[i - 1]) /
                             (self.saved_s[i] - self.saved_s[i - 1]))
                        Fs[j] = p * self.saved_f[i] + (
                            1.0 - p) * self.saved_f[i - 1]
                        value_set = True
                    i += 1
            return Fs
        else:
            predict_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={'x': np.array([[x] for x in s])},
                num_epochs=1,
                shuffle=False)
            predictions = list(
                self.estimator.predict(input_fn=predict_input_fn))
            predictions = [p["Fs"] for p in predictions]
            return predictions

    def _ObstacleAvoidanceForce(self, x, v, o):
        magnitude = np.linalg.norm(x - o)
        direction = (x - o) / magnitude
        p = self.gamma * direction * np.exp(-(self.beta * magnitude)**2)
        return p

    def Plan(self,
             start,
             initial_velocity,
             goal,
             time,
             resolution,
             obstacles=[[], []]):
        num_samples = int(time / resolution)
        t = np.arange(num_samples) * resolution
        s = np.exp(-self.alpha * t / time)
        x = np.zeros((len(s), 2))
        x_dot = np.zeros((len(s), 2))
        v = np.zeros((len(s), 2))
        v_dot = np.zeros((len(s), 2))

        Fs = self._F(s)

        x[0] = start
        x_dot[0] = initial_velocity

        for i in range(1, len(s)):
            # v_dot[i] = (self.K * (goal - x[i - 1]) - self.D * v[i - 1] - self.K * (goal - x[0]) * s[i] + self.K * Fs[i]) / time
            v_dot[i] = ((goal - x[i - 1]) - v[i - 1] -
                        (goal - x[0]) * s[i] + Fs[i]) / time
            for j in range(len(obstacles[0])):
                p = self._ObstacleAvoidanceForce(
                    np.array(x[i - 1]),
                    np.array(v[i - 1]),
                    np.array([obstacles[0][j], obstacles[1][j]]))
                v_dot[i] += (p / time)
            v[i] = v[i - 1] + v_dot[i]
            x_dot[i] = v[i] / time
            x[i] = x[i - 1] + x_dot[i]
            # v[i] = (v[i - 1] + a[i]) / time
            # x[i] = x[i - 1] + v[i]

        if self.verbose:
            print("(DMP: %s) Distance from goal: %s" %
                  (self.name, np.linalg.norm(goal - x[-1])))
        return (x[:, 0], x[:, 1], t)


def GenerateSineDemonstration(time_in_seconds,
                              distance,
                              resolution,
                              noise=0.0,
                              cycles=2):
    samples = time_in_seconds / resolution
    t = np.arange(samples) * resolution
    x = (t / time_in_seconds * distance + np.random.normal(
        loc=0.0, scale=noise, size=len(t)))
    y = (np.sin(2 * np.pi * cycles * t / samples / resolution) +
         np.random.normal(loc=0.0, scale=noise, size=len(t)))
    return (x, y, t)


def VisualizeLearnedOutput(x,
                           y,
                           t,
                           x_prime,
                           y_prime,
                           t_prime,
                           axis_range=2,
                           obstacles=[[], []],
                           title="no_title",
                           show_plot=False):
    f, ax = plt.subplots(3, 1)
    ax[0].plot(x, y, 'b-')
    ax[0].plot(x_prime, y_prime, 'r.')
    ax[0].plot(obstacles[0], obstacles[1], 'go')
    ax[0].set_title("Trajectory path")

    ax[1].plot(t, x, 'b-')
    ax[1].plot(t_prime, x_prime, 'r.')
    ax[1].set_title("x(t)")

    ax[2].plot(t, y, 'b-')
    ax[2].plot(t_prime, y_prime, 'r.')
    ax[2].set_title("y(t)")

    f.subplots_adjust(hspace=0.3)
    file_name = os.path.join(
        os.path.dirname(__file__), '../images/' + str(title) + '.png')
    f.savefig(file_name)
    if (show_plot):
        plt.show()


def main(args):
    """ Main function. """
    time_in_seconds = 5
    distance = 2
    cycles = 2
    resolution = 0.1
    K = 1000
    obstacles1 = [
        [1.25, 0.5],
        [1.0, 0.0],
    ]
    obstacles2 = [
        [1.0],
        [0.5],
    ]

    # 2 meters in 5 seconds
    (x, y, t) = GenerateSineDemonstration(
        time_in_seconds, distance, resolution, cycles=cycles)

    dmp = DynamicMovementPrimitive2D(
        K, name="dmp", verbose=args.verbose, debug=args.debug)
    dmp.TrainOnce(np.transpose([x, y]), t)

    # 2. Prove that we can relearn the original trajectory.
    start = [0, 0]
    goal = [distance, 0]
    (x_prime, y_prime, t_prime) = dmp.Plan(start, [0, 0], goal, time_in_seconds,
                                           resolution)
    print("Replicate original demonstration.")
    VisualizeLearnedOutput(
        x,
        y,
        t,
        x_prime,
        y_prime,
        t_prime,
        title="linear_demonstration",
        show_plot=False)

    # 3. Test spatial generalization. Demonstrate the learned sine-wave motion
    #    with a significantly different start and goal.
    (x_prime, y_prime, t_prime) = dmp.Plan([-1, 1], [0, 0], [2, 10],
                                           time_in_seconds, resolution)
    print("Generalize to new start and goal.")
    VisualizeLearnedOutput(
        x,
        y,
        t,
        x_prime,
        y_prime,
        t_prime,
        title="linear_generalization_spatial")

    # 4. Test temporal generalization. Repeat the original trajectory at
    #    half-speed and double-speed.

    # Double-speed.
    (x_prime, y_prime, t_prime) = dmp.Plan(start, [0, 0], goal,
                                           time_in_seconds / 2, resolution)
    print("Do original demonstration movement at double-speed.")
    VisualizeLearnedOutput(
        x,
        y,
        t,
        x_prime,
        y_prime,
        t_prime,
        title="linear_generalization_double_speed")

    # Half-speed.
    (x_prime, y_prime, t_prime) = dmp.Plan(start, [0, 0], goal,
                                           time_in_seconds * 2, resolution)
    print("Do original demonstration movement at half-speed.")
    VisualizeLearnedOutput(
        x,
        y,
        t,
        x_prime,
        y_prime,
        t_prime,
        title="linear_generalization_half_speed")

    # 7a. Test obstacle avoidance planning with linear interpolation.
    (x_prime, y_prime, t_prime) = dmp.Plan(start, [0, 0], goal, time_in_seconds,
                                           resolution, obstacles1)
    print("Perform original demonstration with obstacle avoidance.")
    VisualizeLearnedOutput(
        x,
        y,
        t,
        x_prime,
        y_prime,
        t_prime,
        obstacles=obstacles1,
        title="linear_obstacle_avoidance_1")
    (x_prime, y_prime, t_prime) = dmp.Plan(start, [0, 0], goal, time_in_seconds,
                                           resolution, obstacles2)
    VisualizeLearnedOutput(
        x,
        y,
        t,
        x_prime,
        y_prime,
        t_prime,
        obstacles=obstacles2,
        title="linear_obstacle_avoidance_2")

    # 5. Create a replicate demonstration of the original, but with some
    #    gaussian noise.
    (noisy_x, noisy_y, noisy_t) = GenerateSineDemonstration(
        time_in_seconds, distance, resolution, noise=0.03, cycles=cycles)
    print("Create noisy version of original demonstration.")
    VisualizeLearnedOutput(
        x, y, t, noisy_x, noisy_y, noisy_t, title="noisy_demonstration")

    # 6. Train on multiple demonstrations and linearly regress over radial
    #    basis functions. Test how well generalization still works.
    dmp.TrainOnce(np.transpose([noisy_x, noisy_y]), noisy_t)
    (x_prime, y_prime, t_prime) = dmp.Plan(start, [0, 0], goal, time_in_seconds,
                                           resolution)
    print("Perform original demonstration as a regression over both examples.")
    VisualizeLearnedOutput(
        x, y, t, x_prime, y_prime, t_prime, title="regression_demonstration")

    # Test to see if generalization still works on different start and goal
    # locations.
    (x_prime, y_prime, t_prime) = dmp.Plan([-1, 1], [0, 0], [2, 10],
                                           time_in_seconds, resolution)
    print("Generalize to a new start and goal with regression.")
    VisualizeLearnedOutput(
        x,
        y,
        t,
        x_prime,
        y_prime,
        t_prime,
        title="regression_generalization_spatial")

    # 7. Add a coupling term to avoid obstacles.
    (x_prime, y_prime, t_prime) = dmp.Plan(start, [0, 0], goal, time_in_seconds,
                                           resolution, obstacles1)
    print("Avoid obstacles with regression.")
    VisualizeLearnedOutput(
        x,
        y,
        t,
        x_prime,
        y_prime,
        t_prime,
        obstacles=obstacles1,
        title="regression_obstacle_avoidance_1")

    (x_prime, y_prime, t_prime) = dmp.Plan(start, [0, 0], goal, time_in_seconds,
                                           resolution, obstacles2)
    VisualizeLearnedOutput(
        x,
        y,
        t,
        x_prime,
        y_prime,
        t_prime,
        obstacles=obstacles2,
        title="regression_obstacle_avoidance_2")


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
        parser.add_argument(
            '-d',
            '--debug',
            action='store_true',
            default=False,
            help='debug output')
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
