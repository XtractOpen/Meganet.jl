"""Contains resnet model
"""

import tensorflow as tf
import numpy as np
import time 
import math
from utils_input import *
import os


class ResnetModel(object):

    def __init__(self, config):
        self.channels   = config['channels']
        self.k_size     = config['kSize']
        self.batchsize  = config['batchSize']
        self.epochs     = config['maxEpochs']
        self.h          = config['h']
        self.num_units  = config['numUnits']
        self.num_blocks = config['numBlocksPerUnit']
        self.xtrain     = config['xTrain']
        self.xval       = config['xValid']
        self.ytrain     = config['yTrain']
        self.yval       = config['yValid']

        self.initial_channels = 3
        self.num_train = self.ytrain.shape[0]
        self.num_val = self.yval.shape[0]
        self.n_class = 10

        self.strides = [1,1,1,1]
        self.activation_func = tf.tanh
        self.epochs_completed = 0
        self.weight_decay_rate = .0002
        self.lr = .01
        self.momentum = .9
        self.shuffle = True
    
    def train(self):
        with tf.device("/cpu:0"):
            self.X = tf.placeholder(tf.float32, [None, self.xtrain.shape[1]])
            self.y = tf.placeholder(tf.float32, [None, self.n_class])

            self.logits = self.build_model()
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits))
            self.loss += weight_decay(self.weight_decay_rate, keyword=r'K_')

            self.train_op = tf.train.MomentumOptimizer(self.lr, self.momentum).minimize(self.loss)

            # Create init op
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

            # Only run on single thread
            session_conf = tf.ConfigProto(
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=1)

            with tf.Session(config=session_conf) as self.sess:

                # Initialize variables
                self.sess.run(init_op)

                self.train_steps_per_epoch = int(math.ceil(self.num_train / float(self.batchsize)))
                self.test_steps_per_epoch = self.num_val // self.batchsize

                # Time it!
                start_time = time.time()

                train_index = 0
                for ep in range(self.epochs):

                    for s in range(self.train_steps_per_epoch):
                        train_batch, train_index = self.next_train_batch(train_index)
                        train_data = train_batch[0]
                        train_labels = train_batch[1]
                        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={self.X: train_data, self.y: train_labels})
                        print(s)
                        print(loss)

                    self.validate_model()
                    self.epochs_completed += 1

        end_time = time.time()
        print("Total time: %f" % (end_time - start_time))

    def validate_model(self):
        # Run training
        avg_train = 0
        n_ex = min(2**12, self.num_train)
        n_steps = int(math.ceil(n_ex / float(self.batchsize)))
        train_index = 0
        for s in range(n_steps):
            train_batch, train_index = self.next_train_batch(train_index)
            train_data = train_batch[0]
            train_labels = train_batch[1]
            _, loss = self.sess.run([self.train_op, self.loss], feed_dict={self.X: train_data, self.y: train_labels})
            avg_train += loss
        avg_train /= n_steps
        print("overall train:")
        print(avg_train)

        # Run Validation
        val_index = 0
        avg_loss = 0
        for i in range(self.test_steps_per_epoch):
            val_batch, val_index = self.next_val_batch(val_index)
            val_data = val_batch[0]
            val_labels = val_batch[1]

            preds, loss = self.sess.run([self.logits, self.loss], feed_dict={self.X: val_data,
                        self.y: val_labels})

            avg_loss += loss

        avg_loss /= self.test_steps_per_epoch
        print("val: %f" % avg_loss)

    def build_model(self):
        with tf.device("/cpu:0"):
            # Reshape
            X = tf.reshape(self.X, [-1, 32, 32, self.initial_channels])
            # Opening Convolution
            K_opening = weight_variable([self.k_size, self.k_size, self.initial_channels, self.channels[0]], 'K_Opening')
            X = tf.nn.conv2d(X, K_opening, strides=self.strides, padding='SAME')

            # Opening Batch Norm
            X = tf.contrib.layers.batch_norm(X, fused=True, is_training=True, scope='Opening')

            #Opening Activation
            X = self.activation_func(X)

            # Build Units
            for unit_num in range(self.num_units):
                X = self.unit(X, unit_num)

            # Average Pooling Layer
            X = tf.reduce_mean(X, [1,2])
            X = tf.reshape(X, [self.batchsize, -1])

            # Fully Connected Layer
            K_FC = weight_variable([self.channels[-1], self.n_class], 'K_FC')
            B_FC = bias_variable([self.n_class], 'B_FC')

            # Return Logits
            logits = tf.nn.xw_plus_b(X, K_FC, B_FC)
            return logits

    def unit(self, X, unit_num):
        with tf.variable_scope("unit_%d" % unit_num):
            for block_num in range(self.num_blocks[unit_num] - 1):
                X = self.block(X, unit_num, block_num)
            
            # Create Connector Block
            X = self.connector_block(X, unit_num)

            return X

    def block(self, X0, unit_num, block_num):
        with tf.variable_scope("block_%d" % block_num):
            # First Convolution
            K1 = weight_variable([self.k_size, self.k_size, self.channels[unit_num], self.channels[unit_num]], 'K_1')
            X1 = tf.nn.conv2d(X0, K1, strides=self.strides, padding='SAME')
        
            # Batch Norm 
            X1 = tf.contrib.layers.batch_norm(X1, fused=True, is_training=True, scope='bn_1')

            # Activation
            X1 = self.activation_func(X1)

            # Second Convolution
            X1 = -tf.nn.conv2d_transpose(X1, K1,
                                  output_shape=tf.concat([tf.shape(X1)[0:3], [tf.shape(K1)[2]]], axis=0),
                                  strides=self.strides,
                                  padding='SAME')
            X1 = tf.reshape(X1, tf.shape(X0))

            # Second Batch Norm
            X1 = tf.contrib.layers.batch_norm(X1, fused=True, is_training=True, scope='bn_2')

            # Compute return value of resnet block
            X = X0 + self.h[unit_num] * X1

            return X

    def connector_block(self, X, unit_num):
        with tf.variable_scope("connector_block"):
            # Convolution increasing channels
            K_Conn = weight_variable([1, 1, self.channels[unit_num], self.channels[unit_num + 1]], 'K_Conn')
            X = tf.nn.conv2d(X, K_Conn, strides=self.strides, padding='SAME')

            # Batch Norm
            X = tf.contrib.layers.batch_norm(X, fused=True, is_training=True, scope='connector')

            # Activation
            X = self.activation_func(X)

            X = tf.nn.avg_pool(X, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1], padding='VALID')

            return X
    
    def next_train_batch(self, index):
        """Gets the next batch of data for training
        
        Arguments:
            index {int} -- Current index in the training data 
        
        Returns:
            batch {tuple of numpy arrays} -- First np array is the batch of data and second is batch of labels
            new_index {int} -- The new index in the training data after creating batch
        """

        start_index = index
        end_index = index + self.batchsize
        final_index = self.num_train - 1

        # First batch
        if self.shuffle and self.epochs_completed == 0 and start_index == 0:
            self.shuffle_data()
        
        # If we are reaching the end of an epoch
        if final_index < end_index:

            # Get remaining data in epoch
            remainder = end_index - final_index
            x_remaining = self.xtrain[start_index:final_index]
            y_remaining = self.ytrain[start_index:final_index]

            if self.shuffle: 
                self.shuffle_data()
            
            # Get new data in next epoch
            x_extra = self.xtrain[0:remainder]
            y_extra = self.ytrain[0:remainder]

            # Combine data into single batch
            x_batch = np.concatenate((x_remaining, x_extra), axis=0)
            y_batch = np.concatenate((y_remaining, y_extra), axis=0)
            batch = (x_batch, y_batch)

            # # Increment epochs_completed
            # self.epochs_completed += 1

            new_index = remainder
        else:
            batch = (self.xtrain[start_index:end_index], self.ytrain[start_index:end_index])
            new_index = end_index

        return batch, new_index

    def shuffle_data(self):
        """Shuffles the training data
        """
        perm = np.arange(self.num_train)
        np.random.shuffle(perm)
        self.xtrain = self.xtrain[perm]
        self.ytrain = self.ytrain[perm]

    def next_val_batch(self, index):
        """Gets the next batch of data for validation
        
        Arguments:
            index {int} -- Current index in the validation data
        
        Returns:
            batch {tuple of numpy arrays} -- First np array is the batch of data and second is batch of labels
            new_index {int} -- The new index in the validation data after creating batch
        """
        if self.epochs_completed == 0 and index == 0:
            # Shuffle validation data at start
            perm = np.arange(self.num_val)
            np.random.shuffle(perm)
            self.xval = self.xval[perm]
            self.yval = self.yval[perm]
        #TODO: Ensure we are validating on all of the validation data
        start_index = index
        end_index = index + self.batchsize

        batch = (self.xval[start_index:end_index], self.yval[start_index:end_index])
        new_index = end_index

        return batch, new_index