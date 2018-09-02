# DeepLabV3 Semantic Segmentation
Reimplementation of DeepLabV3 Semantic Segmentation

This is an (re-)implementation of [DeepLabv3 -- Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587) in TensorFlow for semantic image segmentation on the [PASCAL VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/). The implementation is based on [DrSleep's implementation on DeepLabV2](https://github.com/DrSleep/tensorflow-deeplab-resnet) and [CharlesShang's implementation on tfrecord](https://github.com/CharlesShang/FastMaskRCNN).

## Features
- [x] Tensorflow support
- [ ] Multi-GPUs on single machine (synchronous update)
- [ ] Multi-GPUs on multi servers (asynchronous update)
- [x] ImageNet pre-trained weights
- [ ] Pre-training on MS COCO
- [x] Evaluation on VOC 2012
- [ ] Multi-scale evaluation on VOC 2012

## Requirement
#### Tensorflow 1.4
```
python 3.5
tensorflow 1.4
CUDA  8.0
cuDNN 6.0
```

#### Tensorflow 1.2
```
python 3.5
tensorflow 1.2
CUDA  8.0
cuDNN 5.1
```
The code written in Tensorflow 1.4 are compatible with Tensorflow 1.2, tested on single GPU machine.

#### Installation
```
sh setup.sh
```

## Train
1. Configurate `config.py`.
2. Run `python3 convert_voc12.py --split-name=SPLIT_NAME`, this will generate a tfrecord file in `$DATA_DIRECTORY/records`.
3. Single GPU: Run `python3 train_voc12.py` (with validation mIOU every SAVE_PRED_EVERY).
