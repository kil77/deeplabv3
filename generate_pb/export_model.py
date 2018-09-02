import tensorflow as tf

from PIL import Image
from config import *
from tensorflow.python.framework import graph_util
from libs.nets import deeplabv3

pb_path = args.snapshot_dir

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
        ckpt_path = ckpt_dir + '/model.ckpt-%i' % args.ckpt
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    image = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 3], name='image')

    net, end_points = deeplabv3(image, num_classes=args.num_classes, depth=args.num_layers, is_training=False)

    raw_output = end_points['resnet_v1_{}/logits'.format(args.num_layers)]
    with tf.variable_scope('inference_out'):
        seg_pred = tf.argmax(raw_output, axis=3)
    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session(config=config) as sess:

        image_data = np.array(Image.open('./datasets/IDImage/JPGImage/id00001.jpg'))
        image_data = np.expand_dims(image_data,0)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        restore_var = [v for v in tf.global_variables()]
        loader = tf.train.Saver(var_list=restore_var)
        load(loader, sess, args.snapshot_dir)

        run_ops = [seg_pred];output_node_names = ['inference_out/ArgMax']
        sess.run(run_ops, feed_dict={image: image_data,})

        output_graph_def = graph_util.convert_variables_to_constants(sess, input_graph_def,output_node_names)

        with tf.gfile.GFile(os.path.join(pb_path, "model.pb"), "wb") as f:
            f.write(output_graph_def.SerializeToString())

if __name__ == '__main__':
    main()