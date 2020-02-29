'''
    Single-GPU training.
    Will use H5 dataset in default. If using normal, will shift to the normal dataset.
'''
import argparse
import math
from datetime import datetime
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import tf_util
import modelnet_dataset
import modelnet_h5_dataset
import provider_pu as data_provider

os.environ['CUDA_VISIBLE_DEVICES'] = '9'
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='autoencoder_hierarchical_self_attention', help='Model name [default: pointrnn_cls_basic]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=500, help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='rmsprop, adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--normal', action='store_true', help='Whether to use normal information')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

# ASSIGN_MODEL_PATH=None
# USE_DATA_NORM = True
# USE_RANDOM_INPUT = True
# USE_REPULSION_LOSS = True

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99


HOSTNAME = socket.gethostname()


# Shapenet official train/test split

if FLAGS.normal:
    assert(NUM_POINT<=10000)
    DATA_PATH = os.path.join(ROOT_DIR, 'data/modelnet40_normal_resampled')
    TRAIN_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='train', normal_channel=False, modelnet10=True, batch_size=BATCH_SIZE, unsupervised=True)
    TEST_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='test', normal_channel=False, modelnet10=True, batch_size=BATCH_SIZE)
else:
    assert(NUM_POINT<=2048)
    TRAIN_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'), batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=True)
    TEST_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'), batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=False)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    # learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def ModelStatistics():
    from functools import reduce
    size = lambda v: reduce(lambda x, y: x * y, v.get_shape().as_list())
    for v in tf.trainable_variables():
        print(v.name,v.device,size(v),v.dtype)
    print("total model size:", sum(size(v) for v in tf.trainable_variables()))

def generator_noise_distribution(n_samples, ndims, mu, sigma):
    return np.random.normal(mu, sigma, (n_samples, ndims))


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            # pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            # pointclouds_pl_1, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, 10000)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [], initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            pred, end_points, local_region_t, local_region_g, _, _, _, _ = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)

            ModelStatistics()
            # MODEL.get_loss(pred, labels_pl, end_points)
            loss_local_scale1 = MODEL.create_loss(local_region_g[0], local_region_t[0], 'null')
            loss_local_scale2 = MODEL.create_loss(local_region_g[1], local_region_t[1], 'null')
            loss_local_scale3 = MODEL.create_loss(local_region_g[2], local_region_t[2], 'null')
            loss_local_scale4 = MODEL.create_loss(local_region_g[3], local_region_t[3], 'null')
            loss_shape = MODEL.create_loss(pred, pointclouds_pl, 'null')

            total_loss = loss_local_scale1 + loss_local_scale2 + loss_local_scale3 + loss_local_scale4 + loss_shape
            tf.summary.scalar('total_loss', total_loss)

            print "--- Get training operator"
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9) #,beta2=0.9)
                # optimizer = tf.train.AdamOptimizer(BASE_LEARNING_RATE)
            elif OPTIMIZER == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(learning_rate)
            train_op = optimizer.minimize(total_loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver(max_to_keep=1000)

        # Create a session
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        ops = {'pointclouds_pl': pointclouds_pl,
               # 'pointclouds_pl_1': pointclouds_pl_1,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': total_loss,
               'loss_shape': loss_shape,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            train_one_epoch(sess, ops, train_writer, test_writer)
            if (epoch+1) % 100 == 0:
                saver.save(sess, os.path.join(LOG_DIR, "model"+str(epoch+1)+".ckpt"))

def train_one_epoch(sess, ops, train_writer, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = True

    log_string(str(datetime.now()))

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,TRAIN_DATASET.num_channel()))
    # cur_batch_data1 = np.zeros((BATCH_SIZE, 10000, TRAIN_DATASET.num_channel()))
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)

    loss_sum = 0
    batch_idx = 0
    loss_s1_sum = 0
    loss_s2_sum = 0
    loss_s3_sum = 0
    loss_s4_sum = 0
    loss_shape_sum = 0
    while TRAIN_DATASET.has_next_batch():
    # for batch_idx in range(fetchworker.num_batches):

        batch_data, batch_label = TRAIN_DATASET.next_batch(augment=False)
        bsize = batch_data.shape[0]
        cur_batch_data[0:bsize, ...] = batch_data
        # cur_batch_data1[0:bsize, ...] = batch_data
        cur_batch_label[0:bsize] = batch_label

        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     # ops['pointclouds_pl_1']: cur_batch_data1,
                     ops['labels_pl']: cur_batch_label,
                     ops['is_training_pl']: is_training}
        summary, step, _, loss_val = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        loss_sum += loss_val
        batch_idx += 1
        if batch_idx % 10 == 0:
            print('show total_loss: %f' % (
            loss_sum / float(batch_idx)))
            # print('show loss_s1: %f, loss_s2: %f, loss_s3: %f, loss_s4: %f, loss_shape: %f, total_loss: %f' % (loss_s1_sum / float(batch_idx),
            #                                                                                                  loss_s2_sum / float( batch_idx),
            #                                                                                                  loss_s3_sum / float(batch_idx),
            #                                                                                                  loss_s4_sum / float(batch_idx),
            #                                                                                                  loss_shape_sum / float(batch_idx),
            #                                                                                                  loss_sum / float(batch_idx)))
    # log_string('eval eopch loss_s1: %f, loss_s2: %f, loss_s3: %f, loss_s4: %f, loss_shape: %f, total_loss: %f' % (loss_s1_sum / float(batch_idx),
    #                                                                                                  loss_s2_sum / float( batch_idx),
    #                                                                                                  loss_s3_sum / float(batch_idx),
    #                                                                                                  loss_s4_sum / float(batch_idx),
    #                                                                                                  loss_shape_sum / float(batch_idx),
    #                                                                                                  loss_sum / float(batch_idx)))
    log_string('eval eopch total_loss: %f' % (loss_sum / float(batch_idx)))
    # log_string('show scale4_loss: %f , total_loss: %f' % (
        # loss_s1_sum / float(batch_idx),
        # loss_s2_sum / float(batch_idx),
        # loss_s3_sum / float(batch_idx),
        # loss_s4_sum / float(batch_idx),
        # loss_sum / float(batch_idx)))
    TRAIN_DATASET.reset()


def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,TEST_DATASET.num_channel()))
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)

    loss_sum = 0
    batch_idx = 0


    log_string(str(datetime.now()))
    #log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))

    while TEST_DATASET.has_next_batch():
        batch_data, batch_label = TEST_DATASET.next_batch(augment=False)
        bsize = batch_data.shape[0]
        # for the last batch in the epoch, the bsize:end are from last batch
        cur_batch_data[0:bsize,...] = batch_data
        cur_batch_label[0:bsize] = batch_label

        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['labels_pl']: cur_batch_label,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['pred']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        loss_sum += loss_val
        batch_idx += 1


    log_string('eval mean test loss: %f' % (loss_sum / float(batch_idx)))
    EPOCH_CNT += 1

    TEST_DATASET.reset()


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
