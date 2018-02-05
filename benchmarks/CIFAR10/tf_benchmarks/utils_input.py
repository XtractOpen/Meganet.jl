"""Contains helper functions for running tensorflow benchmark
"""

import numpy as np
import os
import sys
from six.moves import urllib
from six.moves import cPickle
import tarfile
import tensorflow as tf

def one_hot(labels, n_class):
    """ Return one hot encoding of labels

    Args:
        labels: a vector of length n, with values in [0, n_class-1]
        n_class: number of classes, typically 10
    Returns:
    One hot encoding, a (n, n_class) np array.
    """

    return np.eye(n_class)[labels].reshape((labels.shape[0], n_class))

def maybe_download_and_extract(DATA_URL, dest_directory, extracted_filepath=None):
    """Download and extract the tarball from Alex's website."""
    # https://github.com/tensorflow/models/blob/dac6755b121f1446ec857cd05c2ff53b2fd26b90/tutorials/image/cifar10/cifar10.py

    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    if extracted_filepath:
        extracted_dir_path = os.path.join(dest_directory, extracted_filepath)
        if not os.path.exists(extracted_dir_path):
            tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def load_batch(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data.
    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.
    # Returns
        A tuple `(data, labels)`.
    """
    # https://github.com/fchollet/keras/blob/3e933ca0ed1c526c0a9b8643ca84129db96ecc17/keras/datasets/cifar.py

    f = open(fpath, 'rb')
    if sys.version_info < (3,):
        d = cPickle.load(f)
    else:
        d = cPickle.load(f, encoding='bytes')
        # decode utf8
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode('utf8')] = v
        d = d_decoded
    f.close()
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32).transpose([0, 2, 3, 1])
    return data, labels

def load_cifar10(dirname, ntrain, nval):
    """Loads CIFAR10 dataset.
    # Returns
        Tuple of np arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    CIFAR10_DATA_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"   
     
    maybe_download_and_extract(
        CIFAR10_DATA_URL, dirname, extracted_filepath='cifar-10-batches-py')
    path = os.path.join(dirname, 'cifar-10-batches-py')

    nb_train_samples = 50000

    x_train = np.zeros((nb_train_samples, 32, 32, 3), dtype='uint8')
    y_train = np.zeros((nb_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        data, labels = load_batch(fpath)
        x_train[(i - 1) * 10000: i * 10000, :, :, :] = data
        y_train[(i - 1) * 10000: i * 10000] = labels

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    x_train = x_train.reshape((x_train.shape[0], 32 * 32 * 3))
    x_test = x_test.reshape((x_test.shape[0], 32 * 32 * 3))

    y_train_one_hot = one_hot(y_train, 10)
    y_test_one_hot = one_hot(y_test, 10)

    return (x_train[:ntrain], y_train_one_hot[:ntrain]), (x_test[:nval], y_test_one_hot[:nval])

def weight_decay(weight_decay_rate, keyword):
    """Adds weight decay using l2 loss to all variables with names matching keyword
    
    Arguments:
        weight_decay_rate {float} -- The decay rate factor
        keyword {string or regex} --  The keyword to match variable names to
    
    Returns:
        [Tensor] -- A Tensor containing the total loss from weight decay
    """
    costs = []
    for var in tf.trainable_variables():
        if var.op.name.find(keyword) > 0:
            costs.append(tf.nn.l2_loss(var) / 2)

    if len(costs) == 0:
        # No variables match the keyword
        return 0
    else:
        return tf.multiply(weight_decay_rate, tf.add_n(costs))

def weight_variable(shape, name):
    """Creates a tensorflow variable with given shape and name initialized using Xavier Initialization
    Arguments:
        shape {array} -- Array with the dimensions needed for the variable
        name {string} -- Name for the variable
    
    Returns:
        [tf.Variable] -- A trainable tensorflow variable of type float32
    """
    return tf.get_variable(name, shape=shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

def bias_variable(shape, name):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.get_variable(name, dtype=tf.float32, initializer=initial)