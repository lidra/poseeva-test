# from keras.models import Sequential
import random

from keras.layers.core import Flatten
from keras.layers.core import Dropout, Reshape, Lambda
from keras.layers.convolutional import Conv2D, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import tensorflow as tf
# from keras import backend as K
# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import cv2
import os
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys

import numpy as np
import tensorflow as tf
import cv2
from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization
BIN, OVERLAP = 2, 0.1
W = 1.
ALPHA = 1.
MAX_JIT = 3
NORM_H, NORM_W = 224, 224
# VEHICLES = ['Car', 'Truck', 'Van', 'Tram']
VEHICLES = ['Cyclist', 'Van', 'Tram', 'Car', 'Misc', 'Pedestrian', 'Truck', 'Person_sitting']
PASCAL = ['aeroplane', 'bus', 'sofa', 'train', 'boat', 'bottle', 'car', 'chair', 'motorbike', 'tvmonitor', 'bicycle',
          'diningtable']  # 'DontCare']
VEHICLES.extend(PASCAL)
BATCH_SIZE = 8

hdf5path = 'hdf5/kitti-weights-nodim.hdf5'

slim = tf.contrib.slim

sys.path.append('../')

c2int = {}
int2c = {}
ssdlabel = open('ssdlabel.txt', 'r')
for line in ssdlabel.readlines():
    c = line.split(' ')[0]
    i = line.split(' ')[1].strip()
    c2int[c] = i
    int2c[i] = c
ssdlabel.close()
# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

# Input placeholder.
net_shape = (300, 300)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
ckpt_filename = './checkpoints/ssd_300_vgg.ckpt'
# ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)


# Main image processing routine.
def process_image(imgpath, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
    img = mpimg.imread(imgpath)
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        rpredictions, rlocalisations, ssd_anchors,
        select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes


# Test on some demo image and visualize output.
path = './demo/'
image_names = sorted(os.listdir(path))


def getbbox(img):
    img = mpimg.imread(img)
    rclasses, rscores, rbboxes = process_image(img)
    # visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
    visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
    return rclasses, rbboxes


# img = './demo/000001.jpg'
# rclass, bbox = getbbox(img)
# print bbox

def l2_normalize(x):
    return tf.nn.l2_normalize(x, dim=2)

def plt_bboxes_save(imgpath, classes, scores, bboxes, figsize=(10,10), linewidth=1.5):
    """Visualize bounding boxes. Largely inspired by SSD-MXNET!
    """
    img = mpimg.imread(imgpath)
    fig = plt.figure(figsize=figsize)
    plt.imshow(img)
    height = img.shape[0]
    width = img.shape[1]
    colors = dict()
    for i in range(classes.shape[0]):
        cls_id = int(classes[i])
        if cls_id >= 0:
            score = scores[i]
            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(), random.random())
            ymin = int(bboxes[i, 0] * height)
            xmin = int(bboxes[i, 1] * width)
            ymax = int(bboxes[i, 2] * height)
            xmax = int(bboxes[i, 3] * width)
            rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                 ymax - ymin, fill=False,
                                 edgecolor=colors[cls_id],
                                 linewidth=linewidth)
            plt.gca().add_patch(rect)
            class_name = int2c[str(cls_id)]
            plt.gca().text(xmin, ymin - 2,
                           '{:s} | {:.3f}'.format(class_name, score),
                           bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                           fontsize=12, color='white')
    plt.savefig(os.path.splitext(imgpath)[0]+'-2dbox'+os.path.splitext(imgpath)[1])
    plt.savefig(os.path.splitext(imgpath)[0]+'-2dbox'+os.path.splitext(imgpath)[1])

for im in os.listdir(path):
# imgpath = './demo/000001.jpg'
    imgpath = os.path.join(path, im)
    rclasses,rscores, bboxs = process_image(imgpath)
    plt_bboxes_save(imgpath, rclasses, rscores, bboxs)
    # Construct the network
    inputs = Input(shape=(224, 224, 3))
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten()(x)

    # dimension = Dense(512)(x)
    # dimension = LeakyReLU(alpha=0.1)(dimension)
    # dimension = Dropout(0.5)(dimension)
    # dimension = Dense(3)(dimension)
    # dimension = LeakyReLU(alpha=0.1, name='dimension')(dimension)

    orientation = Dense(256)(x)
    orientation = LeakyReLU(alpha=0.1)(orientation)
    orientation = Dropout(0.5)(orientation)
    orientation = Dense(BIN * 2)(orientation)
    orientation = LeakyReLU(alpha=0.1)(orientation)
    orientation = Reshape((BIN, -1))(orientation)
    orientation = Lambda(l2_normalize, name='orientation')(orientation)

    confidence = Dense(256)(x)
    confidence = LeakyReLU(alpha=0.1)(confidence)
    confidence = Dropout(0.5)(confidence)
    confidence = Dense(BIN, activation='softmax', name='confidence')(confidence)

    model = Model(inputs, outputs=[orientation, confidence])
    model.load_weights(hdf5path)

    # groundtruth.append(line2[3])
    # containimg.append(image_file)

    # truncated = np.abs(float(line[1]))
    # occluded = np.abs(float(line[2]))
    img = cv2.imread(imgpath)
    for i in range(bboxs.shape[0]):
        bbox = bboxs[i]
        rclass = rclasses[i]
        print int2c[str(rclass)]
        height = img.shape[0]
        width = img.shape[1]
        ymin = int(bbox[0] * height)
        xmin = int(bbox[1] * width)
        ymax = int(bbox[2] * height)
        xmax = int(bbox[3] * width)
        xmin = max(0, xmin)  # + np.random.randint(-MAX_JIT, MAX_JIT+1)
        ymin = max(0, ymin)  # + np.random.randint(-MAX_JIT, MAX_JIT+1)
        xmax = min(xmax, img.shape[1])  # + np.random.randint(-MAX_JIT, MAX_JIT+1)
        ymax = min(ymax, img.shape[0])  # + np.random.randint(-MAX_JIT, MAX_JIT+1)
        print xmin,ymin,xmax,ymax
        patch = img[ymin:ymax, xmin:xmax]
        patch = cv2.resize(patch, (NORM_H, NORM_W))
        patch = patch - np.array([[[103.939, 116.779, 123.68]]])
        patch = np.expand_dims(patch, 0)
        # print 'patch: '+str(patch)

        prediction = model.predict(patch)

        # print prediction
        # Transform regressed angle
        max_anc = np.argmax(prediction[1][0])

        anchors = prediction[0][0][max_anc]
        # print 'abchors: '+str(anchors)

        if anchors[1] > 0:
            angle_offset = np.arccos(anchors[0])
        else:
            angle_offset = -np.arccos(anchors[0])

        wedge = 2. * np.pi / BIN
        angle_offset = angle_offset + max_anc * wedge
        angle_offset = angle_offset % (2. * np.pi)

        angle_offset = angle_offset - np.pi / 2
        if angle_offset > np.pi:
            angle_offset = angle_offset - (2. * np.pi)
        print angle_offset
