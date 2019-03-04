import tensorflow as tf
from utils.flags import FLAGS
import utils.helper as helper
import utils.tree_util as tree_util
import numpy as np
import sys
import os
import shutil
import webbrowser
import time

#config
sentence_embedding_size = 4
word_embedding_size = 3
label_size = 2

class word_embed_util():
    embeddings = [[0.11, 0.12, 0.13],
                  [0.21, 0.22, 0.23],
                  [0.31, 0.32, 0.33]]


class Data():
    def __init__(self):
        self.train_trees = tree_util.parse_trees("test")
        self.test_trees = tree_util.parse_trees("test")
        self.val_trees = tree_util.parse_trees("test")

        self.word_embed_util = word_embed_util()


data = Data()
# constants
embeddings = tf.constant(data.word_embed_util.embeddings)
## dummi values
rep_zero = tf.constant(0., shape=[sentence_embedding_size, 1])
word_zero = tf.constant(0., shape=[word_embedding_size, 1])
label_zero = tf.constant(0., shape=[label_size, 1])

# tree structure placeholders
root_array = tf.placeholder(tf.int32, (None), name='root_array')
is_leaf_array = tf.placeholder(tf.bool, (None, None), name='is_leaf_array')
word_index_array = tf.placeholder(tf.int32, (None, None), name='word_index_array')
left_child_array = tf.placeholder(tf.int32, (None, None), name='left_child_array')
right_child_array = tf.placeholder(tf.int32, (None, None), name='right_child_array')
label_array = tf.placeholder(tf.int32, (None, None, FLAGS.label_size), name='label_array')

# initializers
xavier_initializer = tf.contrib.layers.xavier_initializer()