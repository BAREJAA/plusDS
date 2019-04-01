from __future__ import division
import tensorflow as tf
import os
import pandas as pd
import numpy as np
from nets import inception_utils, inception_v3
from glob import glob
import imageio
import sys
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

def get_checkpoint_init_fn():
    # Load from .ckpt file
    variables_to_restore = slim.get_variables_to_restore(exclude=["InceptionV3/Logits/Conv2d_1c_1x1/weights:0", "InceptionV3/Logits/Conv2d_1c_1x1/biases:0"])
    global_step_reset = tf.assign(tf.train.get_or_create_global_step(), 0)
    slim_init_fn = slim.assign_from_checkpoint_fn("./inception_v3.ckpt",variables_to_restore,ignore_missing_vars=True)
    return slim_init_fn

# Load Data
if sys.platform == "darwin":
    base_skin_dir = os.path.join('./Data/')
else:
    base_skin_dir = os.path.join('/datacommons/plusds/skin_cancer/team2')
slim = tf.contrib.slim

# Set Params
MAX_EPOCH = 50000
NUM_CLASSES = 7
NUM_IMG_FROM_EACH_CLASS = 9
input_size = NUM_IMG_FROM_EACH_CLASS * NUM_CLASSES
VALIDATION_INTERVAL = 500
START_LR = 1e-03
DECAY_STEP = 10000 / 63 * 2.5
DECAY_RATE = 0.94
LOG_DIR = "./saved_model/Inception_" + str(START_LR) + "_" + str(DECAY_STEP) + "_" + str(DECAY_RATE)

session_config = tf.ConfigProto(log_device_placement=False)
session_config.gpu_options.allow_growth = True

session_config = tf.ConfigProto(log_device_placement=False)
session_config.gpu_options.allow_growth = True

# Dictionary for Loading Resized Images
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(base_skin_dir, 'HAM10000_images_part_[1-2]_resize', '*.jpg'))}

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'dermatofibroma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

# Side Information DataFrame

side_df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'))
side_df['path'] = side_df['image_id'].map(imageid_path_dict.get)
side_df['cell_type'] = side_df['dx'].map(lesion_type_dict.get) 
side_df['cell_type_idx'] = pd.Categorical(side_df['cell_type']).codes

image_by_type = [side_df.iloc[np.array(side_df["cell_type_idx"] == i)] for i in range(len(lesion_type_dict))]
image_by_type_train = [i.head(int(np.ceil(len(i)*0.9))) for i in image_by_type]
image_by_type_val = [i.tail(int(np.floor(len(i)*0.1))) for i in image_by_type]

# Train
one_hot_encoder = OneHotEncoder(NUM_CLASSES)
one_hot_encoder.fit(np.arange(NUM_CLASSES).reshape(-1,1))

g = tf.Graph()

with g.as_default():
    img_holder = tf.placeholder(shape=[input_size,299,299,3], dtype=tf.float32, name="Img_Holder")
    label_holder = tf.placeholder(shape=[input_size,7], dtype=tf.float32, name="Label_Holder")
    img = tf.Variable(img_holder, name="Img_Var", trainable=False)
    label = tf.Variable(label_holder, name="Label_Var", trainable=False)
    img_assign = img.assign(img_holder, name="Img_Assign")
    label_assign = label.assign(label_holder, name="Label_Assign")
    
    img_holder_val = tf.placeholder(shape=[input_size,299,299,3], dtype=tf.float32, name="Img_Holder_val")
    label_holder_val = tf.placeholder(shape=[input_size,7], dtype=tf.float32, name="Label_Holder_val")
    img_val = tf.Variable(img_holder_val, name="Img_Var_val", trainable=False)
    label_val = tf.Variable(label_holder_val, name="Label_Var_val", trainable=False)
    img_assign_val = img_val.assign(img_holder_val, name="Img_Assign_val")
    label_assign_val = label_val.assign(label_holder_val, name="Label_Assign_val")
#     dataset = tf.data.Dataset.from_tensor_slices((img_holder,label_holder)).batch(input_size)
#     iterator = dataset.make_initializable_iterator()
#     image, label = iterator.get_next()
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, end_points = inception_v3.inception_v3(img, num_classes=7, create_aux_logits=False, is_training=True)
        logits_val, _ = inception_v3.inception_v3(img_val, num_classes=7, create_aux_logits=False, is_training=False, reuse=tf.AUTO_REUSE)
    loss = tf.losses.softmax_cross_entropy(label, logits)
    total_loss = tf.losses.get_total_loss()
    loss_val = tf.losses.softmax_cross_entropy(label_val, logits_val, loss_collection="validation")
    
    learning_rate = tf.train.exponential_decay(START_LR, tf.train.get_or_create_global_step(), DECAY_STEP, DECAY_RATE)
    opt = tf.train.GradientDescentOptimizer(learning_rate)
    train_tensor = slim.learning.create_train_op(total_loss, optimizer=opt)
    # Creat Summary
    slim.summaries.add_scalar_summary(total_loss, 'cross_entropy_loss', 'losses')
    slim.summaries.add_scalar_summary(learning_rate, 'learning_rate', 'training')
    slim.summaries.add_scalar_summary(loss_val, 'validation_loss', 'losses')
    slim.summaries.add_scalar_summary(loss_val-total_loss, 'validation_delta', 'losses')
    

def train_step_fn(sess, train_op, global_step, train_step_kwargs):
    """
    slim.learning.train_step():
    train_step_kwargs = {summary_writer:, should_log:, should_stop:}
    """
#     train_step_fn.step += 1  # or use global_step.eval(session=sess)
    input_path = np.array([image_by_type_train[i]["path"].sample(NUM_IMG_FROM_EACH_CLASS) for i in range(NUM_CLASSES)]).reshape(-1)
    input_images = np.array([imageio.imread(i) for i in input_path]).astype(np.float32)
    labels = np.array([[i]*NUM_IMG_FROM_EACH_CLASS for i in range(NUM_CLASSES)]).reshape(-1,1)
    input_images, labels = shuffle(input_images, labels)
    labels = one_hot_encoder.transform(labels).toarray()
    
    sess.run([img_assign,label_assign], feed_dict={img_holder:input_images, label_holder:labels})
#     print sess.run([img,label])

    # calc training losses
    total_loss, should_stop = slim.learning.train_step(sess, train_op, global_step, train_step_kwargs)


    # validate on interval
    if global_step.eval(session=sess) % VALIDATION_INTERVAL == 0:
        input_path_val = np.array([image_by_type_val[i]["path"].sample(NUM_IMG_FROM_EACH_CLASS) for i in range(NUM_CLASSES)]).reshape(-1)
        input_images_val = np.array([imageio.imread(i) for i in input_path_val]).astype(np.float32)
        labels_val = np.array([[i]*NUM_IMG_FROM_EACH_CLASS for i in range(NUM_CLASSES)]).reshape(-1,1)
        input_images_val, labels_val = shuffle(input_images_val, labels_val)
        labels_val = one_hot_encoder.transform(labels_val).toarray()
        
        sess.run([img_assign_val,label_assign_val,logits_val], feed_dict={img_holder_val:input_images_val, label_holder_val:labels_val})
        validiate_loss = sess.run(loss_val)

        
        print(">> global step {}:    train={}   validation={}  delta={}".format(global_step.eval(session=sess), 
                            total_loss, validiate_loss, validiate_loss-total_loss))


    return [total_loss, should_stop]

with g.as_default():
        
        
#     tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, )
    
    input_path = np.array([image_by_type_train[i]["path"].sample(NUM_IMG_FROM_EACH_CLASS) for i in range(NUM_CLASSES)]).reshape(-1)
    input_images = np.array([imageio.imread(i) for i in input_path]).astype(np.float32)
#     input_size = np.shape(input_images)[0]
    labels = np.array([[i]*NUM_IMG_FROM_EACH_CLASS for i in range(NUM_CLASSES)]).reshape(-1,1)
    input_images, labels = shuffle(input_images, labels)
    labels = one_hot_encoder.transform(labels).toarray()
    
    input_path_val = np.array([image_by_type_val[i]["path"].sample(NUM_IMG_FROM_EACH_CLASS) for i in range(NUM_CLASSES)]).reshape(-1)
    input_images_val = np.array([imageio.imread(i) for i in input_path_val]).astype(np.float32)
    labels_val = np.array([[i]*NUM_IMG_FROM_EACH_CLASS for i in range(NUM_CLASSES)]).reshape(-1,1)
    input_images_val, labels_val = shuffle(input_images_val, labels_val)
    labels_val = one_hot_encoder.transform(labels_val).toarray()

    slim.learning.train(
        train_tensor,
        LOG_DIR,
        log_every_n_steps=1,
        number_of_steps=MAX_EPOCH,
        graph=g,
        save_summaries_secs=60,
        save_interval_secs=300,
        init_fn=get_checkpoint_init_fn(),
        global_step=tf.train.get_global_step(),
        train_step_fn = train_step_fn,
        session_config=session_config,
        init_feed_dict = {img_holder:input_images, label_holder:labels, img_holder_val: input_images_val, label_holder_val: labels_val})

    