'''

Create on December 4, 2018

@author: Xinhai Liu

'''

import tensorflow as tf
import numpy as np
import os
import sys
from tflearn.layers.conv import conv_1d, avg_pool_1d
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.core import fully_connected, dropout
BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, '../../models/ups'))
from tf_utils import expand_scope_by_name, replicate_parameter_for_all_layers


sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../../utils'))
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module
sys.path.append(os.path.join(BASE_DIR, '../../latent_3d_points/external/structural_losses'))
import tf_approxmatch
import tf_nndistance

def encoder_with_convs_and_symmetry(in_signal, n_filters=[64, 128, 256, 1024], filter_sizes=[1], strides=[1],
                                        b_norm=True, non_linearity=tf.nn.relu, regularizer=None, weight_decay=0.001,
                                        symmetry=tf.reduce_max, dropout_prob=None, pool=avg_pool_1d, pool_sizes=None, scope=None,
                                        reuse=False, padding='same', verbose=False, closing=None, conv_op=conv_1d):
    '''An Encoder (recognition network), which maps inputs onto a latent space.
    '''

    if verbose:
        print 'Building Encoder'

    n_layers = len(n_filters)
    filter_sizes = replicate_parameter_for_all_layers(filter_sizes, n_layers)
    strides = replicate_parameter_for_all_layers(strides, n_layers)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

    if n_layers < 2:
        raise ValueError('More than 1 layers are expected.')

    for i in xrange(n_layers):
        if i == 0:
            layer = in_signal

        name = 'encoder_conv_layer_' + str(i)
        scope_i = expand_scope_by_name(scope, name)
        layer = conv_op(layer, nb_filter=n_filters[i], filter_size=filter_sizes[i], strides=strides[i], regularizer=regularizer,
                        weight_decay=weight_decay, name=name, reuse=reuse, scope=scope_i, padding=padding)

        if verbose:
            print name, 'conv params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),

        if b_norm:
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
            if verbose:
                print 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list())

        if non_linearity is not None:
            layer = non_linearity(layer)

        if pool is not None and pool_sizes is not None:
            if pool_sizes[i] is not None:
                layer = pool(layer, kernel_size=pool_sizes[i])

        if dropout_prob is not None and dropout_prob[i] > 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])

        if verbose:
            print layer
            print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'

    if symmetry is not None:
        layer = symmetry(layer, axis=1)
        if verbose:
            print layer

    if closing is not None:
        layer = closing(layer)
        print layer

    return layer

def decoder_with_upsameple(point_cloud, l1_xyz, l1_points, l2_xyz, l2_points, l3_xyz, l3_points, l4_xyz, l4_points, is_training, bn_decay=None):

    n_points = point_cloud.get_shape()[1].value
    end_points = {}

    l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256], is_training, bn_decay, scope='fa_layer1')

    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256, 256], is_training, bn_decay, scope='fa_layer2')

    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256, 128], is_training, bn_decay, scope='fa_layer3')

    l0_points = pointnet_fp_module(point_cloud, l1_xyz, None, l1_points, [128, 128], is_training, bn_decay, scope='fa_layer4')

    # Fully connected layers
    out_signal = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    out_signal = tf_util.conv1d(out_signal, 64, 1, padding='VALID', bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    end_points['feats'] = out_signal
    # out_signal = tf_util.dropout(out_signal, keep_prob=0.5, is_training=is_training, scope='dp1')
    out_signal = tf_util.conv1d(out_signal, 3, 1, padding='VALID', activation_fn=None, scope='fc3')

    return out_signal, end_points


def decoder_with_fc_only(latent_signal, layer_sizes=[], b_norm=True, non_linearity=tf.nn.relu,
                         regularizer=None, weight_decay=0.001, reuse=False, scope=None, dropout_prob=None,
                         b_norm_finish=False, verbose=False):
    '''A decoding network which maps points from the latent space back onto the data space.
    '''
    if verbose:
        print 'Building Decoder'

    n_layers = len(layer_sizes)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

    if n_layers < 2:
        raise ValueError('For an FC decoder with single a layer use simpler code.')

    for i in xrange(0, n_layers - 1):
        name = 'decoder_fc_' + str(i)
        scope_i = expand_scope_by_name(scope, name)

        if i == 0:
            layer = latent_signal

        layer = fully_connected(layer, layer_sizes[i], activation='linear', weights_init='xavier', name=name, regularizer=regularizer, weight_decay=weight_decay, reuse=reuse, scope=scope_i)

        if verbose:
            print name, 'FC params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),

        if b_norm:
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
            if verbose:
                print 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list())

        if non_linearity is not None:
            layer = non_linearity(layer)

        if dropout_prob is not None and dropout_prob[i] > 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])

        if verbose:
            print layer
            print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'

    # Last decoding layer never has a non-linearity.
    name = 'decoder_fc_' + str(n_layers - 1)
    scope_i = expand_scope_by_name(scope, name)
    layer = fully_connected(layer, layer_sizes[n_layers - 1], activation='linear', weights_init='xavier', name=name, regularizer=regularizer, weight_decay=weight_decay, reuse=reuse, scope=scope_i)
    if verbose:
        print name, 'FC params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),

    if b_norm_finish:
        name += '_bnorm'
        scope_i = expand_scope_by_name(scope, name)
        layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
        if verbose:
            print 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list())

    if verbose:
        print layer
        print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'

    return layer

def decoder_with_convs_only(in_signal, n_filters, filter_sizes, strides, padding='same', b_norm=True, non_linearity=tf.nn.relu,
                            conv_op=conv_1d, regularizer=None, weight_decay=0.001, dropout_prob=None, upsample_sizes=None,
                            b_norm_finish=False, scope=None, reuse=False, verbose=False):

    if verbose:
        print 'Building Decoder'

    n_layers = len(n_filters)
    filter_sizes = replicate_parameter_for_all_layers(filter_sizes, n_layers)
    strides = replicate_parameter_for_all_layers(strides, n_layers)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

    for i in xrange(n_layers):
        if i == 0:
            layer = in_signal

        name = 'decoder_conv_layer_' + str(i)
        scope_i = expand_scope_by_name(scope, name)

        layer = conv_op(layer, nb_filter=n_filters[i], filter_size=filter_sizes[i],
                        strides=strides[i], padding=padding, regularizer=regularizer, weight_decay=weight_decay,
                        name=name, reuse=reuse, scope=scope_i)

        if verbose:
            print name, 'conv params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),

        if (b_norm and i < n_layers - 1) or (i == n_layers - 1 and b_norm_finish):
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
            if verbose:
                print 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list())

        if non_linearity is not None and i < n_layers - 1:  # Last layer doesn't have a non-linearity.
            layer = non_linearity(layer)

        if dropout_prob is not None and dropout_prob[i] > 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])

        if upsample_sizes is not None and upsample_sizes[i] is not None:
            layer = tf.tile(layer, multiples=[1, upsample_sizes[i], 1])

        if verbose:
            print layer
            print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'

    return layer

def create_loss(output, truth, loss_type = 'emd'):
    if loss_type == 'emd':
        match = tf_approxmatch.approx_match(output, truth)
        build_loss = tf.reduce_mean(tf_approxmatch.match_cost(output, truth, match))

    else:
        cost_p1_p2, _, cost_p2_p1, _ = tf_nndistance.nn_distance(output, truth)
        build_loss = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)
    tf.summary.scalar('build loss', build_loss)
    tf.add_to_collection('losses', build_loss)
    return build_loss