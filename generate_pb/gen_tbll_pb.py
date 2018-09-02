import io
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as pyplot
import tensorflow as tf
import time
import cv2

from PIL import Image
from config import *
from datetime import datetime
from libs.datasets.dataset_factory import read_data
from libs.datasets.VOC12 import decode_labels, inv_preprocess, prepare_label
from libs.nets import deeplabv3
import matplotlib.pyplot as plt
from tensorflow.python.framework import graph_util

slim = tf.contrib.slim
streaming_mean_iou = tf.contrib.metrics.streaming_mean_iou

def save(saver, sess, logdir, step):
   '''Save weights.
   
   Args:
     saver: TensorFlow Saver object.
     sess: TensorFlow session.
     logdir: path to the snapshots directory.
     step: current training step.
   '''
   model_name = 'model.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)
    
   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step)
   print('The checkpoint has been created.')

def load(saver, sess, ckpt_dir):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow Saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    if args.ckpt == 0:
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        ckpt_path = ckpt.model_checkpoint_path
    else:
        ckpt_path = ckpt_dir+'/model.ckpt-%i' % args.ckpt
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def resize_img_and_label(img,label,h,w):
    img = tf.image.resize_bilinear(img, [h, w])
    label = tf.image.resize_nearest_neighbor(label,[h, w])
    return img,label

def get_clipped_image_shape(shape, max_size, min_size, base_size, filename=None):
    """ calculate the image shape to be resized for evaluation
        Args:
            shape: the original image shape
            max_size: the maximum image size
            min_size: the minnum image size
            base_size: the base size of the image, that means the length of any image size should be dividable
                       by this value.
        Returns: the calculated shape that must meet below two conditions,
            1. min_size < width, and height < FLAGS.max_size
                the maximum size is restrained by the GPU memory, and the minimum size
                is for good detection performance
            2. width, and height % base_size == 0, this is required by FCN
    """
    img_h, img_w = shape[0], shape[1]
    min_side, max_side = min(img_h, img_w), max(img_h, img_w)
    aspect_ratio = img_w * 1.0 / img_h
    base = base_size

    # print 'Image: %s' % filename

    def ensure_size(value):
        """ Ensure the image size,
            1. dividable by the base
            2. within the range (FLAGS.min_eval_img_size, FLAGS.max_eval_img_size)
        """
        return int(round(value * 1.0 / base)) * base

    if max_side > max_size:
        if img_h > img_w:
            clipped_w = ensure_size(max_size * aspect_ratio)
            clipped_shape = (max_size, clipped_w)
        else :
            clipped_h = ensure_size(max_size / aspect_ratio)
            clipped_shape = (clipped_h, max_size)
    elif min_side < min_size:
        if img_h < img_w:
            clipped_w = ensure_size(min_size * aspect_ratio)
            clipped_shape = (min_size, clipped_w)
        else:
            clipped_h = ensure_size(min_size / aspect_ratio)
            clipped_shape = (clipped_h, min_size)
    else:
        clipped_h = ensure_size(img_h)
        clipped_shape = (clipped_h, ensure_size(clipped_h * aspect_ratio))

    if clipped_shape[0] < 3 or clipped_shape[1] < 3:
        return None

    return np.asarray(clipped_shape, np.int32)

def main(snapshot_dir, image_path, output_path):
    """Create the model and start the training."""
    tf.set_random_seed(args.random_seed)
    
    # Create queue coordinator.
    coord = tf.train.Coordinator()

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    image = tf.placeholder(dtype=tf.int32, shape=[None, None, 3], name='image')

    image_float = tf.to_float(image)
    image_reverse = tf.reverse(image_float, axis=[-1])
    image_norm = image_reverse - IMG_MEAN
    image_batch = tf.expand_dims(image_norm, axis=0)

    # Create network.
    net, end_points = deeplabv3(image_batch,
                                num_classes=args.num_classes,
                                depth=args.num_layers,
                                is_training=False)

    restore_var = [v for v in tf.global_variables() if 'fc' not in v.name or not args.not_restore_last]
    
    # Predictions.
    raw_output = end_points['resnet_v1_{}/logits'.format(args.num_layers)]
    out_put_score = tf.nn.softmax(raw_output, axis=3)
    out_put_score = tf.reduce_max(out_put_score,axis=3)
    seg_pred = tf.argmax(raw_output, axis=3, output_type=tf.int32)

    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    # Load variables if the checkpoint is provided.
    # if args.ckpt > 0 or args.restore_from is not None:
    loader = tf.train.Saver(var_list=restore_var)
    load(loader, sess, snapshot_dir)

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    tf.get_default_graph().finalize()

    file_list = os.listdir(image_path)
    # image_file = file_list[0]
    image_file = '/home/gq/xnetsrc/image/id00018.jpg'
    file_path = os.path.join(image_path,image_file)
    image_data = cv2.imread(file_path, cv2.IMREAD_COLOR)
    _, seg = sess.run([out_put_score, seg_pred], {image: image_data})

    output_node_names = ['Max','ArgMax']
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, input_graph_def,
        output_node_names
    )

    pb_file = os.path.join(snapshot_dir, "model.pb")
    with tf.gfile.GFile(pb_file, "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("%d ops in the final graph." % len(output_graph_def.node))

    with tf.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    def print_ops():
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name="prefix",
                op_dict=None,
                producer_op_list=None
            )

        for op in graph.get_operations():
            print(op.name, op.values())

    print_ops()

    if coord.should_stop():
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    root_dir = '/home/gq/PycharmProjects/DeepLabV3-Tensorflow-master'
    snapshot_dir = os.path.join(root_dir, 'snapshots')
    image_path = os.path.join(root_dir, 'datasets/IDImage/JPGImage')
    output_path = os.path.join(root_dir, 'datasets/IDImage/inference_out')

    main(snapshot_dir, image_path, output_path)
