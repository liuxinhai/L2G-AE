### L2G Auto-encoder: *Understanding Point Clouds by Local-to-Global Reconstruction with Hierarchical Self-Attention*
Created by Xinhai Liu, Zhizhong Han, Xin Wen, <a href="http://cgcad.thss.tsinghua.edu.cn/liuyushen/" target="_blank">Yu-Shen Liu</a>, Matthias Zwicker.

![prediction example](https://github.com/liuxinhai/L2G-AE/blob/master/doc/L2G-AE.jpg)

### Citation
If you find our work useful in your research, please consider citing:

        @inproceedings{liu2019l2gautoencoder,
          title={ L2G Auto-encoder: Understanding Point Clouds by Local-to-Global Reconstruction with Hierarchical Self-Attention},
          author={Liu, Xinhai and Han, Zhizhong and Wen, Xin and Liu, Yu-Shen and Zwicker, Matthias},
          booktitle={Proceedings of the 27th ACM International Conference on Multimedia},
          year={2019}
        }

### Introduction
In L2G-AE, we focus on learning the local and global structures of point clouds in an auto-encoder architecture.
Specifically, we propose hierarchical self-attentions to learn the correlation among point features in different semantic levels by highlight the importance of each element.
In addition, we also introduce a RNN docoding layer to decode the features of different scale areas in the local region reconstruction. 

In this repository we release code our L2G-AE classification as well as a few utility scripts for training, testing and data processing.

### Installation

Install <a href="https://www.tensorflow.org/install/">TensorFlow</a>. The code is tested under TF1.4 GPU version and Python 2.7 on Ubuntu 16.04. There are also some dependencies for a few Python libraries for data processing like `cv2`, `h5py` etc. It's highly recommended that you have access to GPUs.
Before running the code, you need to compile customized TF operators as described in <a href="https://github.com/charlesq34/pointnet2/">PointNet++</a>.
### Usage

#### Shape Classification

To train a Point2Sequence model to classify ModelNet40 shapes (using point clouds with XYZ coordinates):

        python train.py

To see all optional arguments for training:

        python train.py -h

In the training process, we also evaluate the performance the model.

#### Shape Part Segmentation

To train a model to segment object parts for ShapeNet models:

        cd part_seg
        python train.py
#### Prepare Your Own Data
Follow the dataset in PointNet++, you can refer to <a href="https://github.com/charlesq34/3dmodel_feature/blob/master/io/write_hdf5.py">here</a> on how to prepare your own HDF5 files for either classification or segmentation. Or you can refer to `modelnet_dataset.py` on how to read raw data files and prepare mini-batches from them.
### License
Our code is released under MIT License (see LICENSE file for details).

### Related Projects

* <a href="https://arxiv.org/abs/1706.02413" target="_blank">PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space</a> by Qi et al. (NIPS 2017)
* <a href="http://openaccess.thecvf.com/content_cvpr_2018/html/Xie_Attentional_ShapeContextNet_for_CVPR_2018_paper.html" target="_blank">Attentional ShapeContextNet for Point Cloud Recognition</a> by Xie et al. (CVPR 2018)
