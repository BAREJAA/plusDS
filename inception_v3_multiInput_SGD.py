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

slim = tf.contrib.slim
arg_scope = tf.contrib.framework.arg_scope

# Set Params
MAX_EPOCH = 150000
NUM_CLASSES = 7
NUM_IMG_FROM_EACH_CLASS = 9
input_size = NUM_IMG_FROM_EACH_CLASS * NUM_CLASSES
VALIDATION_INTERVAL = 500
START_LR = 1e-04
DECAY_STEP = 10000 / 63 * 2.5
DECAY_RATE = 0.94

if sys.platform == "darwin":
    base_skin_dir = os.path.join('./Data/')
    LOG_DIR = "./saved_model/Inception_SGD_" + str(START_LR) + "_" + str(DECAY_STEP) + "_" + str(DECAY_RATE)
else:
    base_skin_dir = os.path.join('/datacommons/plusds/skin_cancer/team2')
    LOG_DIR = "/work/qg26/saved_model/Inception_SGD_" + str(START_LR) + "_" + str(DECAY_STEP) + "_" + str(DECAY_RATE)


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

side_df = side_df.dropna()
side_df = side_df[side_df["localization"]!="unknown"]
side_df = side_df[side_df["sex"]!="unknown"]

sex_dict = {"male": 0, "female": 1}
all_localization = [i for i in side_df["localization"].unique()]
localization_dict = {}
for i in range(len(all_localization)):
    localization_dict[all_localization[i]] = i
all_age = [i for i in side_df["age"].unique()]
age_dict = {}
for i in range(len(all_age)):
    age_dict[all_age[i]] = i

image_by_type = [side_df.iloc[np.array(side_df["cell_type_idx"] == i)] for i in range(len(lesion_type_dict))]
image_by_type_train = [i.head(int(np.ceil(len(i)*0.8))) for i in image_by_type]
image_by_type_val = [j.head(int(np.ceil(len(j)*0.5))) for j in [i.tail(int(np.floor(len(i)*0.2))) for i in image_by_type] ]
image_by_type_test = [i.tail(int(np.floor(len(i)*0.1))) for i in image_by_type]

def modified_inception_v3(img, local, sex, age, reuse=None, trainable=True):
    
    with tf.variable_scope("Logits", reuse=reuse) as scope:
        with slim.arg_scope([slim.fully_connected], scope="Logits", reuse=reuse, trainable=trainable):
            local_fc = slim.fully_connected(local, 2048, scope= "Local_FC")
            local_logits = slim.fully_connected(local_fc, 7, scope="Local_Logtis", activation_fn=None, normalizer_fn=None)
            sex_fc = slim.fully_connected(sex, 2048, scope= "Sex_FC")
            sex_logits = slim.fully_connected(sex_fc, 7, scope="Sex_Logtis", activation_fn=None, normalizer_fn=None)
            age_fc = slim.fully_connected(age, 2048, scope= "Age_FC")
            age_logits = slim.fully_connected(age_fc, 7, scope="Age_Logtis", activation_fn=None, normalizer_fn=None)

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, end_points = inception_v3.inception_v3(img, num_classes=7, create_aux_logits=False, is_training=trainable, reuse=reuse)
    
    with tf.variable_scope(scope, auxiliary_name_scope=False) as scope1:
        with tf.name_scope(scope1.original_name_scope) as scope2:
            with slim.arg_scope([], scope=scope2, reuse=reuse):
                logits = tf.add(tf.add(tf.add(local_logits,sex_logits),age_logits),logits, name="Logits")
                pred = slim.softmax(logits, scope="Prediction")
    return logits, pred, end_points

# Train
one_hot_encoder = OneHotEncoder(NUM_CLASSES)
one_hot_encoder.fit(np.arange(NUM_CLASSES).reshape(-1,1))

sex_encoder = OneHotEncoder(len(sex_dict))
sex_encoder.fit(np.arange(len(sex_dict)).reshape(-1,1))

local_encoder = OneHotEncoder(len(localization_dict))
local_encoder.fit(np.arange(len(localization_dict)).reshape(-1,1))

age_encoder = OneHotEncoder(len(age_dict))
age_encoder.fit(np.arange(len(age_dict)).reshape(-1,1))

g = tf.Graph()

with g.as_default():
    img_holder = tf.placeholder(shape=[input_size,299,299,3], dtype=tf.float32, name="Img_Holder")
    label_holder = tf.placeholder(shape=[input_size,7], dtype=tf.float32, name="Label_Holder")
    img = tf.Variable(img_holder, name="Img_Var", trainable=False)
    label = tf.Variable(label_holder, name="Label_Var", trainable=False)
    img_assign = img.assign(img_holder, name="Img_Assign")
    label_assign = label.assign(label_holder, name="Label_Assign")
    
    local_holder = tf.placeholder(shape=[input_size,len(localization_dict)], dtype=tf.float32, name="Local_Holder")
    local = tf.Variable(local_holder, name="Local_Var", trainable=False)
    local_assign = local.assign(local_holder, name="Local_Assign")
    sex_holder = tf.placeholder(shape=[input_size,len(sex_dict)], dtype=tf.float32, name="Sex_Holder")
    sex = tf.Variable(sex_holder, name="Sex_Var", trainable=False)
    sex_assign = sex.assign(sex_holder, name="Sex_Assign")
    age_holder = tf.placeholder(shape=[input_size,len(age_dict)], dtype=tf.float32, name="Age_Holder")
    age = tf.Variable(age_holder, name="Age_Var", trainable=False)
    age_assign = age.assign(age_holder, name="Age_Assign")
    
    img_holder_val = tf.placeholder(shape=[input_size,299,299,3], dtype=tf.float32, name="Img_Holder_val")
    label_holder_val = tf.placeholder(shape=[input_size,7], dtype=tf.float32, name="Label_Holder_val")
    img_val = tf.Variable(img_holder_val, name="Img_Var_val", trainable=False)
    label_val = tf.Variable(label_holder_val, name="Label_Var_val", trainable=False)
    img_assign_val = img_val.assign(img_holder_val, name="Img_Assign_val")
    label_assign_val = label_val.assign(label_holder_val, name="Label_Assign_val")
    
    local_holder_val = tf.placeholder(shape=[input_size,len(localization_dict)], dtype=tf.float32, name="Local_Holder")
    local_val = tf.Variable(local_holder_val, name="Local_Var_val", trainable=False)
    local_assign_val = local_val.assign(local_holder_val, name="Local_Assign_val")
    sex_holder_val = tf.placeholder(shape=[input_size,len(sex_dict)], dtype=tf.float32, name="Sex_Holder")
    sex_val = tf.Variable(sex_holder_val, name="Sex_Var_val", trainable=False)
    sex_assign_val = sex_val.assign(sex_holder_val, name="Sex_Assign_val")
    age_holder_val = tf.placeholder(shape=[input_size,len(age_dict)], dtype=tf.float32, name="Age_Holder")
    age_val = tf.Variable(age_holder_val, name="Age_Var_val", trainable=False)
    age_assign_val = age_val.assign(age_holder_val, name="Age_Assign_val")

#     with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
#         logits, end_points = inception_v3.inception_v3(img, num_classes=7, create_aux_logits=False, is_training=True)
#         logits_val, _ = inception_v3.inception_v3(img_val, num_classes=7, create_aux_logits=False, is_training=False, reuse=tf.AUTO_REUSE)
    logits, pred, end_points = modified_inception_v3(img, local, sex, age)
    logits_val, pred_val, end_points_val = modified_inception_v3(img_val, local_val, sex_val, 
                                                                 age_val, trainable=False, 
                                                                 reuse=tf.AUTO_REUSE)
    _, accuracy_val = tf.metrics.accuracy(tf.math.argmax(pred_val, axis=1), tf.math.argmax(label_val, axis=1))
    
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
    slim.summaries.add_scalar_summary(accuracy_val, 'validation_accuracy', 'accuracy')
    

def train_step_fn(sess, train_op, global_step, train_step_kwargs):
    """
    slim.learning.train_step():
    train_step_kwargs = {summary_writer:, should_log:, should_stop:}
    """
#     train_step_fn.step += 1  # or use global_step.eval(session=sess)
    input_df = [image_by_type_train[i].sample(NUM_IMG_FROM_EACH_CLASS) for i in range(NUM_CLASSES)]
    input_path = np.array([input_df[i]["path"] for i in range(NUM_CLASSES)]).reshape(-1)
    input_images = np.array([imageio.imread(i) for i in input_path]).astype(np.float32)
    input_sex = np.array([input_df[i]["sex"] for i in range(NUM_CLASSES)]).reshape(-1)
    input_local = np.array([input_df[i]["localization"] for i in range(NUM_CLASSES)]).reshape(-1)
    input_age = np.array([input_df[i]["age"] for i in range(NUM_CLASSES)]).reshape(-1)

    labels = np.array([[i]*NUM_IMG_FROM_EACH_CLASS for i in range(NUM_CLASSES)]).reshape(-1,1)
    ages = [[age_dict[i]] for i in input_age]
    sexes = [[sex_dict[i]] for i in input_sex]
    localizations = [[localization_dict[i]] for i in input_local]

    input_images, labels, ages, sexes, localizations = shuffle(input_images, labels, ages, sexes, localizations)
    labels = one_hot_encoder.transform(labels).toarray()
    ages = age_encoder.transform(ages).toarray()
    sexes = sex_encoder.transform(sexes).toarray()
    localizations = local_encoder.transform(localizations).toarray()
    
    sess.run([img_assign,label_assign,sex_assign,local_assign,age_assign], feed_dict={img_holder:input_images, label_holder:labels, local_holder: localizations, sex_holder: sexes, age_holder: ages})
#     print sess.run([img,label])

    # calc training losses
    total_loss, should_stop = slim.learning.train_step(sess, train_op, global_step, train_step_kwargs)


    # validate on interval
    if global_step.eval(session=sess) % VALIDATION_INTERVAL == 0:
        input_df_val = [image_by_type_val[i].sample(NUM_IMG_FROM_EACH_CLASS) for i in range(NUM_CLASSES)]
        input_path_val = np.array([input_df_val[i]["path"].sample(NUM_IMG_FROM_EACH_CLASS) for i in range(NUM_CLASSES)]).reshape(-1)
        input_images_val = np.array([imageio.imread(i) for i in input_path_val]).astype(np.float32)
        input_sex_val = np.array([input_df_val[i]["sex"] for i in range(NUM_CLASSES)]).reshape(-1)
        input_local_val = np.array([input_df_val[i]["localization"] for i in range(NUM_CLASSES)]).reshape(-1)
        input_age_val = np.array([input_df_val[i]["age"] for i in range(NUM_CLASSES)]).reshape(-1)

        labels_val = np.array([[i]*NUM_IMG_FROM_EACH_CLASS for i in range(NUM_CLASSES)]).reshape(-1,1)
        ages_val = [[age_dict[i]] for i in input_age_val]
        sexes_val = [[sex_dict[i]] for i in input_sex_val]
        localizations_val = [[localization_dict[i]] for i in input_local_val]

        input_images_val, labels_val, ages_val, sexes_val, localizations_val = shuffle(input_images_val, labels_val, ages_val, sexes_val, localizations_val)
        labels_val = one_hot_encoder.transform(labels_val).toarray()
        ages_val = age_encoder.transform(ages_val).toarray()
        sexes_val = sex_encoder.transform(sexes_val).toarray()
        localizations_val = local_encoder.transform(localizations_val).toarray()
        
        sess.run([img_assign_val,label_assign_val,local_assign_val,sex_assign_val,age_assign_val,logits_val,accuracy_val], 
                 feed_dict={img_holder_val:input_images_val, label_holder_val:labels_val, 
                            local_holder_val:localizations_val, sex_holder_val: sexes_val, age_holder_val:ages_val})
        validiate_loss = sess.run(loss_val)
    
        
#    print(">> global step {}:    train={}   validation={}  delta={}".format(global_step.eval(session=sess), 
#                        total_loss, loss_val, loss_val-total_loss))


    return [total_loss, should_stop]

with g.as_default():
    
    # Train Set    
    input_df = [image_by_type_train[i].sample(NUM_IMG_FROM_EACH_CLASS) for i in range(NUM_CLASSES)]
    input_path = np.array([input_df[i]["path"] for i in range(NUM_CLASSES)]).reshape(-1)
    input_images = np.array([imageio.imread(i) for i in input_path]).astype(np.float32)
    input_sex = np.array([input_df[i]["sex"] for i in range(NUM_CLASSES)]).reshape(-1)
    input_local = np.array([input_df[i]["localization"] for i in range(NUM_CLASSES)]).reshape(-1)
    input_age = np.array([input_df[i]["age"] for i in range(NUM_CLASSES)]).reshape(-1)

    labels = np.array([[i]*NUM_IMG_FROM_EACH_CLASS for i in range(NUM_CLASSES)]).reshape(-1,1)
    ages = [[age_dict[i]] for i in input_age]
    sexes = [[sex_dict[i]] for i in input_sex]
    localizations = [[localization_dict[i]] for i in input_local]

    input_images, labels, ages, sexes, localizations = shuffle(input_images, labels, ages, sexes, localizations)
    labels = one_hot_encoder.transform(labels).toarray()
    ages = age_encoder.transform(ages).toarray()
    sexes = sex_encoder.transform(sexes).toarray()
    localizations = local_encoder.transform(localizations).toarray()
    
    # Val Set
    input_df_val = [image_by_type_val[i].sample(NUM_IMG_FROM_EACH_CLASS) for i in range(NUM_CLASSES)]
    input_path_val = np.array([input_df_val[i]["path"].sample(NUM_IMG_FROM_EACH_CLASS) for i in range(NUM_CLASSES)]).reshape(-1)
    input_images_val = np.array([imageio.imread(i) for i in input_path_val]).astype(np.float32)
    input_sex_val = np.array([input_df_val[i]["sex"] for i in range(NUM_CLASSES)]).reshape(-1)
    input_local_val = np.array([input_df_val[i]["localization"] for i in range(NUM_CLASSES)]).reshape(-1)
    input_age_val = np.array([input_df_val[i]["age"] for i in range(NUM_CLASSES)]).reshape(-1)

    labels_val = np.array([[i]*NUM_IMG_FROM_EACH_CLASS for i in range(NUM_CLASSES)]).reshape(-1,1)
    ages_val = [[age_dict[i]] for i in input_age_val]
    sexes_val = [[sex_dict[i]] for i in input_sex_val]
    localizations_val = [[localization_dict[i]] for i in input_local_val]

    input_images_val, labels_val, ages_val, sexes_val, localizations_val = shuffle(input_images_val, labels_val, ages_val, sexes_val, localizations_val)
    labels_val = one_hot_encoder.transform(labels_val).toarray()
    ages_val = age_encoder.transform(ages_val).toarray()
    sexes_val = sex_encoder.transform(sexes_val).toarray()
    localizations_val = local_encoder.transform(localizations_val).toarray()

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
        init_feed_dict = {img_holder:input_images, label_holder:labels, local_holder: localizations, sex_holder: sexes, age_holder: ages, img_holder_val: input_images_val, label_holder_val: labels_val, local_holder_val:localizations_val, sex_holder_val: sexes_val, age_holder_val:ages_val})
