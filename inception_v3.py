from __future__ import division
import tensorflow as tf
import os
import pandas as pd
import numpy as np
from nets import inception_utils, inception_v3

def get_checkpoint_init_fn():
    # Load from .ckpt file
    variables_to_restore = slim.get_variables_to_restore(exclude=["InceptionV3/Logits/Conv2d_1c_1x1/weights:0", "InceptionV3/Logits/Conv2d_1c_1x1/biases:0"])
    global_step_reset = tf.assign(tf.train.get_or_create_global_step(), 0)
    slim_init_fn = slim.assign_from_checkpoint_fn("./inception_v3.ckpt",variables_to_restore,ignore_missing_vars=True)
    return slim_init_fn

#load data
base_skin_dir = os.path.join('./Data/')
slim = tf.contrib.slim

session_config = tf.ConfigProto(log_device_placement=False)
session_config.gpu_options.allow_growth = True


for tile_df in pd.read_csv(os.path.join(base_skin_dir, 'hmnist_28_28_RGB.csv'), chunksize=320, low_memory=False):
    input_images = np.reshape(tile_df.values[:,:-1],[-1,28,28,3]).astype(np.float32)
    input_labels = tile_df.values[:,-1]
    input_size = np.shape(input_images)[0]

    g = tf.Graph()
    with g.as_default():
        # Split up data into batches
        dataset = tf.data.Dataset.from_tensor_slices((input_images,input_labels)).batch(32)
        iterator = dataset.make_one_shot_iterator()
        image, label = iterator.get_next()
        # image, label = input_images, input_labels
        label = tf.one_hot(label, depth=7)
        # Creat inception_v3
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits, end_points = inception_v3.inception_v3(image, num_classes=7, final_endpoint="Mixed_5d", create_aux_logits=False)
        loss = tf.losses.softmax_cross_entropy(label, logits)
        learning_rate = tf.train.exponential_decay(1e-04, tf.train.get_or_create_global_step(), input_size / 32 * 2.5 , 0.94)
        opt = tf.train.GradientDescentOptimizer(learning_rate)
        train_tensor = slim.learning.create_train_op(loss, optimizer=opt)
        # Creat Summary
        slim.summaries.add_scalar_summary(loss, 'cross_entropy_loss', 'losses')
        slim.summaries.add_scalar_summary(learning_rate, 'learning_rate', 'training')

        # Train
        slim.learning.train(
            train_tensor,
            "./saved_model/",
            log_every_n_steps=300,
            graph=g,
            save_summaries_secs=300,
            save_interval_secs=600,
            init_fn=get_checkpoint_init_fn(),
            global_step=tf.train.get_global_step(),
            session_config=session_config)