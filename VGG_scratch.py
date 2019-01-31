from __future__ import division
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from skimage.io import imread
import skimage
base_skin_dir = os.path.join('Data/')

tile_df = pd.read_csv(os.path.join(base_skin_dir, 'hmnist_28_28_L.csv'))
input_images = np.reshape(tile_df.values[:,:-1],[-1,28,28,1])
labels = tile_df.values[:,-1]

images = tf.placeholder(tf.float32, [None, 28, 28, 1])
y_true = tf.placeholder(tf.int32, [None])
y_true_one_hot = tf.one_hot(y_true, depth=7)

# Define the model
with tf.name_scope('conv1_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 1, 64], dtype=tf.float32, stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope)
    
with tf.name_scope('conv1_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope)
    
pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    
with tf.name_scope('conv2_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope)
    
with tf.name_scope('conv2_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32, stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope)
    
pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    
with tf.name_scope('conv3_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32, stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope)
    
with tf.name_scope('conv3_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope)
    
with tf.name_scope('conv3_3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope)
    
pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
    
with tf.name_scope('conv4_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32, stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(bias, name=scope)
    
with tf.name_scope('conv4_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(bias, name=scope)
    
with tf.name_scope('conv4_3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(bias, name=scope)
    
pool4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

with tf.name_scope('conv5_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv5 = tf.nn.relu(bias, name=scope)
    
with tf.name_scope('conv5_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv5, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv5 = tf.nn.relu(bias, name=scope)
    
with tf.name_scope('conv5_3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv5, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv5 = tf.nn.relu(bias, name=scope)
    
pool5 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')
    
with tf.name_scope('fc_6') as scope:
    flat = tf.reshape(pool5, [-1, 1*1*512])
    weights = tf.Variable(tf.truncated_normal([1*1*512, 4096], dtype=tf.float32, stddev=1e-1), name='weights')
    mat = tf.matmul(flat, weights)
    biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), trainable=True, name='biases')
    bias = tf.nn.bias_add(mat, biases)
    fc6 = tf.nn.relu(bias, name=scope)
    fc6_drop = tf.nn.dropout(fc6, keep_prob=0.5, name='dropout')

with tf.name_scope('fc_7') as scope:
    weights = tf.Variable(tf.truncated_normal([4096, 4096], dtype=tf.float32, stddev=1e-1), name='weights')
    mat = tf.matmul(fc6, weights)
    biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), trainable=True, name='biases')
    bias = tf.nn.bias_add(mat, biases)
    fc7 = tf.nn.relu(bias, name=scope)
    fc7_drop = tf.nn.dropout(fc7, keep_prob=0.5, name='dropout')
    
with tf.name_scope('fc_8') as scope:
    weights = tf.Variable(tf.truncated_normal([4096, 7], dtype=tf.float32, stddev=1e-1), name='weights')
    mat = tf.matmul(fc7, weights)
    biases = tf.Variable(tf.constant(0.0, shape=[7], dtype=tf.float32), trainable=True, name='biases')
    bias = tf.nn.bias_add(mat, biases)
    
with tf.name_scope('softmax') as scope:
    predictions = tf.nn.softmax(bias)

# predictions = bias

loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(y_true_one_hot, predictions))
optimizer = tf.train.AdadeltaOptimizer(0.001).minimize(loss)

MAX_EPOCH = 10000
BATCH_SIZE = 10
CROSS_VALID_SIZE = 1000
SAVE_PATH = "./saved_model/vgg.ckpt"

config=tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for epoch in range(MAX_EPOCH):
        # Test on batch_idx, train on the rest
        if epoch%CROSS_VALID_SIZE == 0:
            batch_idx = np.floor(epoch/CROSS_VALID_SIZE)
            batch_idx = [int(batch_idx*CROSS_VALID_SIZE), int((batch_idx+1)*CROSS_VALID_SIZE)]
            train_batch_image = input_images[np.concatenate([np.arange(batch_idx[0],dtype=np.int), np.arange(batch_idx[1],10015,dtype=np.int)], axis=0),:]
            train_batch_label = labels[np.concatenate([np.arange(batch_idx[0],dtype=np.int), np.arange(batch_idx[1],10015,dtype=np.int)], axis=0)]
            test_batch_image = input_images[np.arange(batch_idx[0],batch_idx[1]),:]
            test_batch_label = labels[batch_idx[0]:batch_idx[1]]
            
        idx = np.random.randint(10015-CROSS_VALID_SIZE,size=BATCH_SIZE)
        train_data = train_batch_image[idx,:]
        train_labels= train_batch_label[idx]
        _, cost = sess.run([optimizer, loss], feed_dict={images:train_data, y_true:train_labels})
        
        if epoch % CROSS_VALID_SIZE == 0:
            true_pred = 0.
            saver.save(sess, SAVE_PATH)
            cross_valid_idx = np.random.randint(CROSS_VALID_SIZE,size=BATCH_SIZE)
            cross_valid_data = test_batch_image[cross_valid_idx,:]
            cross_valid_label = test_batch_label[cross_valid_idx]
            pred = sess.run(predictions, feed_dict={images:train_data})
            for i in range(len(pred)):
                current_pred = np.argmax(pred[i], axis=0)
                if current_pred == cross_valid_label[i]:
                    true_pred += 1
            acc = true_pred/BATCH_SIZE
            with open(".stats.txt","ab") as myfile:
            	myfile.write("Epoch: " + str(int(epoch)), "Cost: ", str(cost), "Accuracy", str(acc))
            print "Epoch: ", epoch, "Cost: ", cost, "Accuracy", acc
    
