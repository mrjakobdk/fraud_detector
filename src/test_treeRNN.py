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
from utils.word_embeddings_util import WordEmbeddingsUtil

class Data():
    def __init__(self):
        self.train_trees = tree_util.parse_trees("test")
        self.test_trees = tree_util.parse_trees("test")
        self.val_trees = tree_util.parse_trees("test")

        if FLAGS.word_embed_mode == '':
            self.word_embed_util = WordEmbeddingsUtil()
        else:
            self.word_embed_util = WordEmbeddingsUtil(mode=FLAGS.word_embed_mode)

class tRNN():
    feed_dict_list = {"val": [], "test": [], "train": []}



    def __init__(self, data):
        # constants
        self.data = data  # TODO: Make data
        self.embeddings = tf.constant(data.word_embed_util.embeddings)
        # constant output
        self.rep_zero = tf.constant(0., shape=[FLAGS.sentence_embedding_size, 1])
        self.word_zero = tf.constant(0., shape=[FLAGS.word_embedding_size, 1])
        self.label_zero = tf.constant(0., shape=[FLAGS.label_size, 1])

        # loss weight constant w>1 more weight on sensitive loss
        self.weight = tf.constant(FLAGS.sensitive_weight)

        # tree structure placeholders
        self.root_array = tf.placeholder(tf.int32, (None), name='root_array')
        self.is_leaf_array = tf.placeholder(tf.bool, (None, None), name='is_leaf_array')
        self.word_index_array = tf.placeholder(tf.int32, (None, None), name='word_index_array')
        self.left_child_array = tf.placeholder(tf.int32, (None, None), name='left_child_array')
        self.right_child_array = tf.placeholder(tf.int32, (None, None), name='right_child_array')
        self.label_array = tf.placeholder(tf.int32, (None, None, FLAGS.label_size), name='label_array')

        # initializers
        xavier_initializer = tf.contrib.layers.xavier_initializer()

        weight_initializer = xavier_initializer
        if FLAGS.weight_initializer == "identity":
            def custom_initializer(shape_list, dtype, partition_info):
                return tf.initializers.identity(gain=0.5)(shape_list, dtype,
                                                          partition_info) + tf.initializers.random_uniform(minval=-0.05,
                                                                                                           maxval=0.05)(
                    shape_list, dtype, partition_info)

            weight_initializer = custom_initializer

        bias_initializer = xavier_initializer
        if FLAGS.bias_initializer == "zero":
            bias_initializer = tf.initializers.zeros()

        # encoding variables
        W = tf.get_variable(name='W', shape=[FLAGS.sentence_embedding_size, FLAGS.word_embedding_size],
                            initializer=weight_initializer)
        self.W = W

        # phrase weights
        U_l = tf.get_variable(name='U_l', shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                              initializer=weight_initializer)
        U_r = tf.get_variable(name='U_r', shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                              initializer=weight_initializer)
        self.U_l = U_l
        self.U_r = U_r
        self.weights = tf.concat([W, U_l, U_r], axis=1)

        # bias
        self.b = tf.get_variable(name='b', shape=[FLAGS.sentence_embedding_size, 1], initializer=bias_initializer)

        # classifier weights
        V = tf.get_variable(name='V', shape=[FLAGS.label_size, FLAGS.sentence_embedding_size],
                            initializer=xavier_initializer)
        b_p = tf.get_variable(name='b_p', shape=[FLAGS.label_size, 1], initializer=bias_initializer)
        self.V = V
        self.b_p = b_p

    def build_feed_dict_batch(self, roots):
        print("Batch size:", len(roots))

        node_list_list = []
        node_to_index_list = []
        for root in roots:
            node_list = []
            tree_util.depth_first_traverse(root, node_list, lambda node, node_list: node_list.append(node))
            node_list_list.append(node_list)
            node_to_index = helper.reverse_dict(node_list)
            node_to_index_list.append(node_to_index)

        feed_dict = {
            self.root_array: [tree_util.size_of_tree(root) for root in roots],
            self.is_leaf_array: helper.lists_pad([
                [False] + [node.is_leaf for node in node_list]
                for node_list in node_list_list], False),
            self.word_index_array: helper.lists_pad([
                [0] + [self.data.word_embed_util.get_idx(node.value) for node in node_list]
                for node_list in node_list_list], 0),
            self.left_child_array: helper.lists_pad([
                [0] + helper.add_one(
                    [node_to_index[node.left_child] if node.left_child is not None else -1 for node in node_list])
                for node_list, node_to_index in zip(node_list_list, node_to_index_list)], 0),
            self.right_child_array: helper.lists_pad([
                [0] + helper.add_one(
                    [node_to_index[node.right_child] if node.right_child is not None else -1 for node in node_list])
                for node_list, node_to_index in zip(node_list_list, node_to_index_list)], 0),
            self.label_array: helper.lists_pad([
                [[0, 0]] + [node.label for node in node_list]
                for node_list in node_list_list], [0, 0])
        }

        print(feed_dict[self.right_child_array])

        return feed_dict

    def test(self):
        pass

data = Data()

treeRNN = tRNN(data)