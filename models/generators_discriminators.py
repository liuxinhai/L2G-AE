'''

Create on December 4, 2018

@author: Xinhai Liu

'''

import tensorflow as tf
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../../utils'))
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module
from encoders_decoders import decoder_with_fc_only, decoder_with_convs_only, encoder_with_convs_and_symmetry
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.core import fully_connected, dropout
BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, '../../models/ups'))
from tf_utils import expand_scope_by_name, replicate_parameter_for_all_layers, leaky_relu

def mlp_discriminator(in_signal, non_linearity=tf.nn.relu, reuse=False, scope=None, b_norm=True, dropout_prob=None):
    ''' used in nips submission.
    '''
    encoder_args = {'n_filters': [64, 128, 256, 256, 512], 'filter_sizes': [1, 1, 1, 1, 1], 'strides': [1, 1, 1, 1, 1]}
    encoder_args['reuse'] = reuse
    encoder_args['scope'] = scope
    encoder_args['non_linearity'] = non_linearity
    encoder_args['dropout_prob'] = dropout_prob
    encoder_args['b_norm'] = b_norm
    layer = encoder_with_convs_and_symmetry(in_signal, **encoder_args)

    name = 'decoding_logits'
    scope_e = expand_scope_by_name(scope, name)
    d_logit = decoder_with_fc_only(layer, layer_sizes=[128, 64, 1], b_norm=b_norm, reuse=reuse, scope=scope_e)
    d_prob = tf.nn.sigmoid(d_logit)
    return d_prob, d_logit

def latent_code_discriminator_two_layers(in_signal, layer_sizes=[512, 512], b_norm=False, non_linearity=tf.nn.relu, reuse=False, scope=None):
    ''' Used in ICML submission.
    '''
    layer_sizes = layer_sizes + [1]
    d_logit = decoder_with_fc_only(in_signal, layer_sizes=layer_sizes, non_linearity=non_linearity, b_norm=b_norm, reuse=reuse, scope=scope)
    d_prob = tf.nn.sigmoid(d_logit)
    return d_prob, d_logit


def point_cloud_generator(z, pc_dims, layer_sizes=[64, 128, 512, 1024], non_linearity=tf.nn.relu, b_norm=False,
                          b_norm_last=False, dropout_prob=None):
    ''' used in nips submission.
    '''

    n_points, dummy = pc_dims
    if (dummy != 3):
        raise ValueError()

    out_signal = decoder_with_fc_only(z, layer_sizes=layer_sizes, non_linearity=non_linearity, b_norm=b_norm)
    out_signal = non_linearity(out_signal)

    if dropout_prob is not None:
        out_signal = dropout(out_signal, dropout_prob)

    if b_norm_last:
        out_signal = batch_normalization(out_signal)

    out_signal = fully_connected(out_signal, np.prod([n_points, 3]), activation='linear', weights_init='xavier')
    out_signal = tf.reshape(out_signal, [-1, n_points, 3])
    return out_signal
def points_generator_patch_based_fc(latent_code, sampled_points=384, feature_dim = 256, rnn_step=4, layer_sizes=[512, 512],is_training=True,
                        s_points=[16,32,64,128], num_point=1024, non_linearity=tf.nn.relu, b_norm=False, bn_decay=False,b_norm_last=False, dropout_prob=None):

    batch_size = latent_code.get_shape()[0]
    local_features = scale_points_generator_fc(latent_code, scale_dim=[sampled_points, feature_dim], layer_sizes=layer_sizes, b_norm=True)

    '''local branch'''
    l1_points = tf_util.rnn_decoder(local_features, feature_dim, rnn_step, scope='decode_layer1', bn=True, is_training=is_training, bn_decay=bn_decay)

    scale1_feature = tf.reshape(l1_points[:, :, 0, :], [-1, 1, feature_dim])
    scale2_feature = tf.reshape(l1_points[:, :, 1, :], [-1, 1, feature_dim])
    scale3_feature = tf.reshape(l1_points[:, :, 2, :], [-1, 1, feature_dim])
    scale4_feature = tf.reshape(l1_points[:, :, 3, :], [-1, 1, feature_dim])

    #global point cloud build
    scale1_points = scale_points_generator_fc(scale1_feature, scale_dim=[s_points[0], 3],
                                              layer_sizes=[128, 256], b_norm=True)
    scale2_points = scale_points_generator_fc(scale2_feature, scale_dim=[s_points[1], 3],
                                              layer_sizes=[256, 256], b_norm=True)
    scale3_points = scale_points_generator_fc(scale3_feature, scale_dim=[s_points[2], 3],
                                              layer_sizes=[256, 256], b_norm=True)
    scale4_points = scale_points_generator_fc(scale4_feature, scale_dim=[s_points[3], 3],
                                              layer_sizes=[256, 512], b_norm=True)

    scale_points = tf.concat([scale1_points, scale2_points, scale3_points, scale4_points], axis=1)
    points_all_features = tf.reshape(scale_points, [batch_size, 1, -1])
    generate_points = scale_points_generator_fc(points_all_features, scale_dim=[num_point, 3], layer_sizes=[1024, 1024], b_norm=True)
    scale_points_generate = [scale1_points, scale2_points, scale3_points, scale4_points]

    return generate_points, scale_points_generate

def scale_points_generator_fc(scale_feature, scale_dim, layer_sizes=[128,128],non_linearity=tf.nn.relu, b_norm=False,
                          b_norm_last=False, dropout_prob=None):
    ''' used in nips submission.
        '''

    n_points, dummy = scale_dim
    # if (dummy != 3):
    #     raise ValueError()

    out_signal = decoder_with_fc_only(scale_feature, layer_sizes=layer_sizes, non_linearity=non_linearity, b_norm=b_norm)
    out_signal = non_linearity(out_signal)

    if dropout_prob is not None:
        out_signal = dropout(out_signal, dropout_prob)

    if b_norm_last:
        out_signal = batch_normalization(out_signal)
    out_signal = fully_connected(out_signal, np.prod([n_points, dummy]), activation='linear', weights_init='xavier')
    out_signal = tf.reshape(out_signal, [-1, n_points, dummy])
    return out_signal

def scale_points_generator_conv(scale_feature, scale_dim, layer_sizes=[128,128],non_linearity=tf.nn.relu, b_norm=False,
                          b_norm_last=False, dropout_prob=None):
    ''' used in nips submission.
        '''

    n_points, dummy = scale_dim
    # if (dummy != 3):
    #     raise ValueError()

    out_signal = decoder_with_convs_only(scale_feature, layer_sizes=layer_sizes, non_linearity=non_linearity, b_norm=b_norm)
    out_signal = non_linearity(out_signal)

    if dropout_prob is not None:
        out_signal = dropout(out_signal, dropout_prob)

    if b_norm_last:
        out_signal = batch_normalization(out_signal)
    out_signal = fully_connected(out_signal, np.prod([n_points, dummy]), activation='linear', weights_init='xavier')
    out_signal = tf.reshape(out_signal, [-1, n_points, dummy])
    return out_signal


def point_cloud_generator(z, pc_dims, layer_sizes=[64, 128, 512, 1024], non_linearity=tf.nn.relu, b_norm=False,
                          b_norm_last=False, dropout_prob=None):
    ''' used in nips submission.
    '''

    n_points, dummy = pc_dims
    if (dummy != 3):
        raise ValueError()

    out_signal = decoder_with_fc_only(z, layer_sizes=layer_sizes, non_linearity=non_linearity, b_norm=b_norm)
    out_signal = non_linearity(out_signal)

    if dropout_prob is not None:
        out_signal = dropout(out_signal, dropout_prob)

    if b_norm_last:
        out_signal = batch_normalization(out_signal)

    out_signal = fully_connected(out_signal, np.prod([n_points, 3]), activation='linear', weights_init='xavier')
    out_signal = tf.reshape(out_signal, [-1, n_points, 3])
    return out_signal