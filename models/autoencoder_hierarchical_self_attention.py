#coding: utf-8
import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import mlp_self_attention, sample_and_group_all, pointnet_fp_module, hierarchical_self_attention
import math
sys.path.append(os.path.join(BASE_DIR, '../tf_ops/sampling'))
from tf_sampling import farthest_point_sample, gather_point
from sklearn.manifold.isomap import Isomap
sys.path.append(os.path.join(BASE_DIR, '../structural_losses'))
import tf_approxmatch
import tf_nndistance
sys.path.append(os.path.join(BASE_DIR, './ups'))
from generators_discriminators import scale_points_generator_fc, point_cloud_generator, scale_points_generator_conv
from encoders_decoders import decoder_with_convs_only



def patch_based_reconstruction(point_cloud, is_training, bn_decay=None):
    """down sampling module"""


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl

def tf_dist(input, batch_size = 16, npoints = 384, channels = 3):
    xyz1 = tf.tile(tf.reshape(input, (batch_size, 1, npoints, channels)), [1, npoints, 1, 1])
    xyz2 = tf.tile(tf.reshape(input, (batch_size, npoints, 1, channels)), [1, 1, npoints, 1])
    dist = tf.reduce_sum((xyz1 - xyz2) ** 2, -1)
    return dist
def get_geo_dis(point_data):
    im = Isomap()
    im.fit(point_data)
    geo_dis = im.dist_matrix_
    return geo_dis

def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    # num_point = 10000
    end_points = {}

    l0_xyz = tf.slice(point_cloud, [0, 0, 0], [-1, -1, 3])
    l0_points = None
    sampled_points = 256
    s1_points = 16
    s2_points = 32
    s3_points = 64
    s4_points = 128
    feature_dim = 256
    shape_feature_dim = 1024
    rnn_step = 4
    attention_dim = 32
    l1_xyz, l1_points, l1_scales, l1_scales_features, pl_attentions, sl_attention = hierarchical_self_attention(l0_xyz, l0_points, sampled_points,
                                                                                   [s1_points, s2_points, s3_points, s4_points],
                                                                                   [[32, feature_dim],
                                                                                    [64, feature_dim],
                                                                                    [64, feature_dim],
                                                                                    [128, feature_dim]], attention_dim, attention_dim,
                                                                                   is_training, bn_decay,
                                                                                   batch_size=batch_size,
                                                                                   scope='down_layer1')

    l4_xyz, latent_representation, l4_indices, rl_attention = mlp_self_attention(l1_xyz, l1_points, npoint=None, ndim=attention_dim,
                                                                   nsample=None, mlp=[256, shape_feature_dim], mlp2=None,
                                                                   group_all=True,
                                                                   is_training=is_training, bn_decay=bn_decay,
                                                                   scope='down_layer3')


    local_features = pointnet_fp_module(l1_xyz, l4_xyz, l1_points, latent_representation, [512, feature_dim], is_training, bn_decay,
                                   scope='fa_layer1')
    '''local branch'''
    l1_points = tf_util.rnn_decoder(local_features, feature_dim, rnn_step, scope='decode_layer1', bn=True, is_training=is_training, bn_decay=bn_decay)
    scale1_feature = tf.reshape(l1_points[:, :, 0, :], [-1, 1, feature_dim])
    scale2_feature = tf.reshape(l1_points[:, :, 1, :], [-1, 1, feature_dim])
    scale3_feature = tf.reshape(l1_points[:, :, 2, :], [-1, 1, feature_dim])
    scale4_feature = tf.reshape(l1_points[:, :, 3, :], [-1, 1, feature_dim])

    # scale1_feature = tf.reshape(l1_points[:, :, 0, :], [-1, 1, 2*feature_dim])
    # scale2_feature = tf.reshape(l1_points[:, :, 1, :], [-1, 1, 2*feature_dim])
    # scale3_feature = tf.reshape(l1_points[:, :, 2, :], [-1, 1, 2*feature_dim])
    # scale4_feature = tf.reshape(l1_points[:, :, 3, :], [-1, 1, 2*feature_dim])
    scale1_points = scale_points_generator_fc(scale1_feature, scale_dim=[s1_points, 3], layer_sizes=[128, 128],
                                              b_norm=True)
    scale2_points = scale_points_generator_fc(scale2_feature, scale_dim=[s2_points, 3], layer_sizes=[128, 128],
                                              b_norm=True)
    scale3_points = scale_points_generator_fc(scale3_feature, scale_dim=[s3_points, 3], layer_sizes=[128, 256],
                                            b_norm=True)
    scale4_points = scale_points_generator_fc(scale4_feature, scale_dim=[s4_points, 3], layer_sizes=[128, 512],
                                              b_norm=True)

    scale_points = tf.concat([scale1_points, scale2_points, scale3_points, scale4_points], axis=1)
    points_all_features = tf.reshape(scale_points, [batch_size, 1, -1])

    generate_points = scale_points_generator_fc(points_all_features, scale_dim=[num_point, 3], layer_sizes=[512, 512],
                                                b_norm=True)

    scale1_xyz = tf.reshape(l1_scales[0], [-1, s1_points, 3])
    scale2_xyz = tf.reshape(l1_scales[1], [-1, s2_points, 3])
    scale3_xyz = tf.reshape(l1_scales[2], [-1, s3_points, 3])
    scale4_xyz = tf.reshape(l1_scales[3], [-1, s4_points, 3])

    scale_points_truth = [scale1_xyz, scale2_xyz, scale3_xyz, scale4_xyz]
    scale_points_generate = [scale1_points, scale2_points, scale3_points, scale4_points]

    end_points['global_feats'] = latent_representation
    return generate_points, end_points, scale_points_truth, scale_points_generate, pl_attentions, sl_attention, rl_attention, l1_xyz


def get_loss(pred, label, end_points):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss

def create_loss(output, truth, loss_type = 'emd'):
    if loss_type == 'emd':
        match = tf_approxmatch.approx_match(output, truth)
        build_loss = tf.reduce_mean(tf_approxmatch.match_cost(output, truth, match))

    else:
        cost_p1_p2, _, cost_p2_p1, _ = tf_nndistance.nn_distance(output, truth)
        build_loss = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)
    # tf.summary.scalar('build loss', build_loss)
    # tf.add_to_collection('losses', build_loss)
    return build_loss

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        net, _ = get_model(inputs, tf.constant(True))
        print(net)
