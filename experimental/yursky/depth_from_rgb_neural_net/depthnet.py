#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

"""
Journey Experimentation: Depth From RGB Data
"""
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import scipy
import sys
import traceback
import tensorflow as tf
import time
import util


# Load the data we are giving you
def load(data_dir, W=320, H=240):
    image_files = glob.glob(data_dir + '/*.jpg')
    label_files = glob.glob(data_dir + '/*.png')
    images = np.array(
        [
            scipy.misc.imresize(scipy.ndimage.imread(image_file), 0.5)
            for image_file in image_files
        ],
        dtype=np.float32) / 255.0
    labels = np.array(
        [
            scipy.misc.imresize(scipy.ndimage.imread(label_file), 0.125)
            for label_file in label_files
        ],
        dtype=np.float32) / 255.0
    cutoff = len(image_files) * 4 // 5
    return images[:cutoff], labels[:cutoff], images[:cutoff], labels[:cutoff]


# Step 1: Augment the training data (try the following, not all might improve the performance)
#  * mirror the image
#  * color augmentations (keep the values to small ranges first then try to expand):
#    - brightness
#    - hue
#    - saturation
#    - contrast
def data_augmentation(I):
    # I = tf.image.random_flip_left_right(I)
    return I


def DrawImages(image, depth, ground_truth, title=None):
    fig = plt.figure(figsize=(15, 5))
    if title:
        fig.suptitle(title, fontsize=16)
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title('Image')
    plt.imshow(image)
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title('Learned Depth Map')
    depth = np.clip(depth, 0, 1)
    plt.imshow(depth, cmap='gray')
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title('Ground Truth Depth Map')
    plt.imshow(ground_truth, cmap='gray')
    if title:
        file_name = os.path.join(
            os.path.dirname(__file__), 'images/' + title + '.png')
        fig.savefig(file_name)
    else:
        plt.show()


def main(args):
    """ Main function. """
    ##################
    # Part 0: Setup. #
    ##################
    image_data, label_data, image_val, label_val = load('data/nyu_datasets')

    W = 320
    H = 240

    print('Input shape: ' + str(image_data.shape))
    print('Labels shape: ' + str(label_data.shape))

    num_classes = 6

    ################################
    # Part 1: Define your ConvNet. #
    ################################
    # Lets clear the tensorflow graph, so that you don't have to restart the
    # notebook every time you change the network.
    tf.reset_default_graph()

    # Set up your input placeholder
    inputs = tf.placeholder(tf.float32, (None, H, W, 3))
    # Set up your input placeholder
    training = tf.placeholder_with_default(False, (), name='training')

    # map_fn applies data_augmentation independently for each image in the
    # batch, since we are not croping let's apply the augmentation before
    # whitening, it does make evaluation easier.
    aug_input = tf.map_fn(data_augmentation, inputs)

    # During evaluation we don't want data augmentation.
    eval_inputs = tf.identity(aug_input, name="inputs")

    # Set up your label placeholders.
    labels = tf.placeholder(tf.float32, (None, H / 4, W / 4), name='labels')

    with tf.name_scope('model'), tf.variable_scope('model'):
        # Step 2: Define the compute graph of your CNN here.
        #     Build the network out of 20 3x3 convolutions without striding and
        #     5 pooling layers with stride=2.
        #     Hint: Use a for loop or two to define the model
        #     Hint: Make sure your classification layer does not have a relu
        #           `activation_fn=None`.
        #   Train this model first.
        # Step 3: Add batch normalization.
        #     Hint: You don't need to use scale or center if you apply BN
        #           before a convolution.
        #     Hint: You do need center if BN is between the conv and a ReLU.
        #     Hint: Don't forget to give the batch_normalization layer a
        #           'training=training' argument.
        #  Train your model (you should see it converge much faster now).
        # Step 4: Add residual connections.
        #     For simplicity you do not need to add a residual connection to
        #     every layer, but add them to at least half of your layers.
        #  Train your model (you should see it converge even faster now).
        coarse = aug_input
        fine = aug_input
        coarse = tf.contrib.layers.conv2d(
            coarse,
            96, [11, 11],
            stride=4,
            weights_regularizer=tf.nn.l2_loss,
            scope="coarse1")
        coarse = tf.contrib.layers.max_pool2d(
            coarse, [2, 2], stride=2, scope="pool1")
        coarse = tf.contrib.layers.conv2d(
            coarse,
            256, [5, 5],
            stride=1,
            weights_regularizer=tf.nn.l2_loss,
            scope="coarse2")
        coarse = tf.contrib.layers.max_pool2d(
            coarse, [2, 2], stride=2, scope="pool2")
        coarse = tf.contrib.layers.conv2d(
            coarse,
            384, [3, 3],
            stride=1,
            weights_regularizer=tf.nn.l2_loss,
            scope="coarse3")
        coarse = tf.contrib.layers.conv2d(
            coarse,
            384, [3, 3],
            stride=1,
            weights_regularizer=tf.nn.l2_loss,
            scope="coarse4")
        coarse = tf.contrib.layers.conv2d(
            coarse,
            256, [3, 3],
            stride=1,
            weights_regularizer=tf.nn.l2_loss,
            scope="coarse5")
        coarse = tf.contrib.layers.conv2d(
            coarse,
            4096, [1, 1],
            stride=1,
            weights_regularizer=tf.nn.l2_loss,
            scope="coarse6")
        coarse = tf.contrib.layers.conv2d_transpose(
            coarse,
            1, [1, 1],
            stride=4,
            weights_regularizer=tf.nn.l2_loss,
            scope="coarse7")
        fine = tf.contrib.layers.conv2d(
            fine,
            63, [9, 9],
            stride=2,
            weights_regularizer=tf.nn.l2_loss,
            scope="fine1")
        fine = tf.contrib.layers.max_pool2d(
            fine, [2, 2], stride=2, scope="pool3")
        fine = tf.concat([fine, coarse], axis=3, name="fine2")
        fine = tf.contrib.layers.conv2d(
            fine,
            64, [5, 5],
            stride=1,
            weights_regularizer=tf.nn.l2_loss,
            scope="fine3")
        fine = tf.contrib.layers.conv2d(
            fine,
            1, [5, 5],
            stride=1,
            weights_regularizer=tf.nn.l2_loss,
            activation_fn=None,
            scope="fine4")
        fine = tf.squeeze(fine, [3])
        loss = tf.reduce_mean(
            tf.losses.mean_squared_error(labels=labels, predictions=fine))

    output = tf.identity(fine, name='output')

    regularization_loss = tf.losses.get_regularization_loss()

    # Let's weight the regularization loss down, otherwise it will hurt the
    # model performance. You can tune this weight if you wish.
    total_loss = loss + 1e-6 * regularization_loss

    # Create an optimizer.
    # NOTE: You might have to play with the learning rate as you try out.
    # batch_normalization (0.001 might work well without BN, 0.1 with, 0.001
    # for resnets)
    optimizer = tf.train.MomentumOptimizer(0.0001, 0.9)

    # Use that optimizer on your loss function (control_dependencies makes sure
    # any batch_norm parameters are properly updated)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        opt = optimizer.minimize(total_loss)

    print("Total number of variables used ",
          np.sum(
              [v.get_shape().num_elements()
               for v in tf.trainable_variables()]), '/', 500000)

    #####################
    # Part 2: Training. #
    #####################
    # Batch size.
    BS = 32

    # Start training session.
    sess = tf.Session()

    # Set up training.
    sess.run(tf.global_variables_initializer())

    # Set up saver.
    saver = tf.train.Saver()
    latest_checkpoint = tf.train.latest_checkpoint('model/')
    if latest_checkpoint:
        starting_epoch = int(re.findall('^.*-([0-9]+)$', latest_checkpoint)[0])
        saver.restore(sess, tf.train.latest_checkpoint('model/'))
    else:
        starting_epoch = 0

    # Train ResNet.
    best_validation = 0.0
    num_epochs = 20
    for epoch in range(starting_epoch + 1, starting_epoch + num_epochs + 1):
        # Let's shuffle the data every epoch.
        np.random.seed(epoch)
        np.random.shuffle(image_data)
        np.random.seed(epoch)
        np.random.shuffle(label_data)
        # Go through the entire dataset once.
        train_losses = []
        for i in range(0, image_data.shape[0] - BS + 1, BS):
            # Train a single batch.
            batch_images, batch_labels = image_data[i:i + BS], label_data[
                i:i + BS]
            train_losses.append(
                sess.run(
                    [total_loss, opt],
                    feed_dict={
                        inputs: batch_images,
                        labels: batch_labels,
                        training: True
                    })[0])

        eval_losses = []
        for i in range(0, image_val.shape[0], BS):
            batch_images, batch_labels = image_val[i:i + BS], label_val[i:
                                                                        i + BS]
            eval_losses.append(
                sess.run(
                    total_loss,
                    feed_dict={inputs: batch_images,
                               labels: batch_labels}))

        sample_image = image_data[0]
        sample_gt = label_data[0]
        sample_depth = sess.run(
            output, feed_dict={inputs: np.stack([sample_image])})[0]
        DrawImages(
            sample_image, sample_depth, sample_gt, title=('epoch%d' % epoch))

        saver.save(sess, 'model/depth-net', global_step=epoch)

        print('[%3d] Train Loss: %0.3f \t Eval Loss: %0.3f' %
              (epoch, np.mean(train_losses), np.mean(eval_losses)))

    # Close training session.
    sess.close()

    #######################
    # Part 3: Evaluation. #
    #######################
    # Start evaluation session.
    sess = tf.Session()
    print('Input shape: ' + str(image_val.shape))
    print('Labels shape: ' + str(label_val.shape))

    eval_losses = []
    for i in range(0, image_val.shape[0], BS):
        batch_images, batch_labels = image_val[i:i + BS], label_val[i:i + BS]
        eval_losses.append(
            sess.run(
                total_loss,
                feed_dict={inputs: batch_images,
                           labels: batch_labels}))
    print("ConvNet Validation Loss: ", np.mean(eval_loss))

    # Close evaluation session.
    sess.close()

    ######################################
    # Part 5 (Optional): See your model. #
    ######################################
    # Show the current graph
    util.show_graph(tf.get_default_graph().as_graph_def())


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
