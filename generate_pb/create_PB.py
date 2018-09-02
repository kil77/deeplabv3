#encoding=utf-8
import tensorflow as tf
from tensorflow.python.tools import freeze_graph

# os.environ['CUDA_VISIBLE_DEVICES']='2'  #设置GPU
model_path = "./snapshots/model.ckpt-264030"  # 设置model的路径


def main():
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph(
        "./snapshots/model.ckpt-264030.meta")
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        # 保存图
        tf.train.write_graph(sess.graph_def, './snapshots', 'tmp_model.pb')
        # 把图和参数结构一起
        freeze_graph.freeze_graph('./snapshots/tmp_model.pb',
                                  '',
                                  False,
                                  model_path,
                                  'Max, ArgMax',
                                  'save/restore_all',
                                  'save/Const:0',
                                  './snapshots/model.pb',
                                  False,
                                  "")
    print("done")


if __name__ == '__main__':
    main()