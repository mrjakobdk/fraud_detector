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


class deepRNN:
    def __init__(self, data):
        """
        :param data: utils.data
        """
        helper._print_header("Constructing deepRNN constants, placeholders and variables")

        # Setup data
        self.data = data  # TODO: Make data
        self.embeddings = tf.constant(data.word_embed_util.embeddings)

        # tree structure placeholders
        # todo maybe needed self.root_array = tf.placeholder(tf.int32, (None), name='root_array')
        self.is_leaf_array = tf.placeholder(tf.bool, (None), name='is_leaf_array')
        self.word_index_array = tf.placeholder(tf.int32, (None), name='word_index_array')
        self.left_child_array = tf.placeholder(tf.int32, (None), name='left_child_array')
        self.right_child_array = tf.placeholder(tf.int32, (None), name='right_child_array')
        self.label_array = tf.placeholder(tf.int32, (None, FLAGS.label_size), name='label_array')

        # ----------------- build variables -----------------
        self.build_variables()

        # ----------------- build tree structure -----------------
        def embed_word(word_index):
            return tf.nn.embedding_lookup(self.embeddings, word_index)

        self.rep_array = [None]*FLAGS.deepRNN_depth
        for layer in range(FLAGS.deepRNN_depth):
            self.rep_array[layer] = tf.TensorArray(
                tf.float32,
                size=0,
                dynamic_size=True,
                clear_after_read=False,
                infer_shape=False)


            def build_node_first(left_child, right_child, rep_array):
                left_is_leaf = tf.gather(self.is_leaf_array, left_child)
                right_is_leaf = tf.gather(self.is_leaf_array, right_child)

                # reshape from vector to matrix with height 300 and width 1
                rep_l = tf.reshape(rep_array.read(left_child), [300, 1])
                rep_r = tf.reshape(rep_array.read(right_child), [300, 1])

                left = tf.cond(
                    left_is_leaf,
                    lambda: tf.matmul(self.W_l, rep_l),
                    lambda: tf.matmul(self.U_l[0], rep_l)
                )

                right = tf.cond(
                    right_is_leaf,
                    lambda: tf.matmul(self.W_r, rep_r),
                    lambda: tf.matmul(self.U_r[0], rep_r)
                )

                return tf.nn.leaky_relu(left + right + self.b_U[i])

            def build_node_rest(left_child, right_child, node_index, rep_array, rep_array_hidden):
                #todo there might be something wrong with using i - because of scooping rules

                # reshape from vector to matrix with height 300 and width 1
                rep_l = tf.reshape(rep_array.read(left_child), [300, 1])
                rep_r = tf.reshape(rep_array.read(right_child), [300, 1])
                rep_h = tf.reshape(rep_array_hidden.read(node_index), [300, 1])
                left = tf.matmul(self.U_l[i], rep_l)
                right = tf.matmul(self.U_r[i], rep_r)
                hidden = tf.matmul(self.H[i], rep_h)
                return tf.nn.leaky_relu(left + right + hidden + self.b_U[i])

            def tree_construction_body(rep_array, i):
                # gather variables
                is_leaf = tf.gather(self.is_leaf_array, i)
                word_index = tf.gather(self.word_index_array, i)
                left_child = tf.gather(self.left_child_array, i)
                right_child = tf.gather(self.right_child_array, i)

                # embed_word = (word_size, 1)
                # build_node = (sent_size , 1)
                rep = tf.cond(
                    is_leaf,
                    lambda: embed_word(word_index),
                    lambda: build_node_rest(left_child, right_child, rep_array)
                )
                rep_array = rep_array.write(i, rep)

                i = tf.add(i, 1)
                return rep_array, i



    def build_variables(self):
        # initializers
        general_initializer, weight_initializer, bias_initializer = self.get_initializers()

        # phrase variables
        self.U_l = [None] * FLAGS.deepRNN_depth
        self.U_r = [None] * FLAGS.deepRNN_depth
        self.b_U = [None] * FLAGS.deepRNN_depth
        for i in range(FLAGS.deepRNN_depth):
            # phrase weights
            self.U_l[i] = tf.get_variable(name='U_l' + str(i),
                                          shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                          initializer=weight_initializer)
            self.U_r[i] = tf.get_variable(name='U_r' + str(i),
                                          shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                          initializer=weight_initializer)
            self.b_U[i] = tf.get_variable(name='b_U' + str(i), shape=[FLAGS.sentence_embedding_size, 1],
                                          initializer=bias_initializer)
        # word variables
        self.W_l = tf.get_variable(name='W_l', shape=[FLAGS.sentence_embedding_size, FLAGS.word_embedding_size],
                                   initializer=weight_initializer)
        self.W_r = tf.get_variable(name='W_r', shape=[FLAGS.sentence_embedding_size, FLAGS.word_embedding_size],
                                   initializer=weight_initializer)
        self.b_W = tf.get_variable(name='b_W', shape=[FLAGS.sentence_embedding_size, 1], initializer=bias_initializer)

        # hidden state variables
        self.H = [None] * FLAGS.deepRNN_depth
        self.b_H = [None] * FLAGS.deepRNN_depth
        self.H_leaf = tf.get_variable(name='H_leaf2',
                                      shape=[FLAGS.sentence_embedding_size, FLAGS.word_embedding_size],
                                      initializer=general_initializer)
        for i in range(1, FLAGS.deepRNN_depth):
            # phrase weights
            self.H = tf.get_variable(name='H_' + str(i),
                                     shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                     initializer=general_initializer)
            self.b_H[i] = tf.get_variable(name='b_H' + str(i), shape=[FLAGS.sentence_embedding_size, 1],
                                          initializer=bias_initializer)
        # classifier variables
        self.V = tf.get_variable(name='V', shape=[FLAGS.label_size, FLAGS.sentence_embedding_size],
                                 initializer=general_initializer)
        self.b_p = tf.get_variable(name='b_p', shape=[FLAGS.label_size, 1], initializer=bias_initializer)

    def get_initializers(self):
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
        return xavier_initializer, weight_initializer, bias_initializer

    def build_first_tree(self):
        pass

    def get_acc(self):
        pass

    def get_loss(self):
        pass

    def train(self):
        pass

    def build_feed_dict_batch(self):
        pass
