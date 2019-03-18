import tensorflow as tf
import utils.helper as helper
from utils.flags import FLAGS
import os
from utils import tree_util, directories, constants
import numpy as np


class treeModel:
    def __init__(self, data, word_embed, model_placement,
                 label_size=FLAGS.label_size,
                 learning_rate=FLAGS.learning_rate):

        # config
        self.data = data
        self.word_embed = word_embed
        self.model_placement = model_placement
        self.learning_rate = learning_rate
        self.label_size = label_size

        self.build_constants()
        self.build_placeholders()
        self.build_variables()
        self.build_model()
        self.build_loss()
        self.build_accuracy()
        self.build_predict()
        self.build_train_op()

    def build_constants(self):
        raise NotImplementedError("Each Model must re-implement this method.")

    def build_placeholders(self):
        raise NotImplementedError("Each Model must re-implement this method.")

    def build_variables(self):
        raise NotImplementedError("Each Model must re-implement this method.")

    def build_model(self):
        raise NotImplementedError("Each Model must re-implement this method.")

    def build_loss(self):
        raise NotImplementedError("Each Model must re-implement this method.")

    def build_accuracy(self):
        raise NotImplementedError("Each Model must re-implement this method.")

    def build_predict(self):
        logits = tf.gather_nd(tf.transpose(self.o_array.stack(), perm=[2, 0, 1]), self.root_array)
        self.labels = tf.gather_nd(self.label_array, self.root_array)
        self.p = tf.nn.softmax(logits, axis=-1)

    def build_train_op(self):
        self.global_step = tf.train.create_global_step()

        if FLAGS.lr_decay > 0:
            n = int(len(self.data.train_trees) / FLAGS.batch_size)
            total_steps = FLAGS.lr_decay * n
            decay_steps = n
            decay_rate = (FLAGS.learning_rate_end / self.learning_rate) ** (decay_steps / total_steps)
            self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, decay_steps,
                                                            decay_rate,
                                                            name='learning_rate') + FLAGS.learning_rate_end

            helper._print_header("Using learning rate with exponential decay")
            helper._print("Decay for every step:", decay_rate)
            helper._print("Learning rate start:", self.learning_rate)
            helper._print("Learning rate end:", FLAGS.learning_rate_end)
            helper._print("2 time end lr after:", FLAGS.lr_decay)
        else:
            self.learning_rate = tf.constant(self.learning_rate)

        if FLAGS.optimizer == constants.ADAM_OPTIMIZER:
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)
        else:
            self.train_op = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss,
                                                                                   global_step=self.global_step)

    def initialize(self, sess):
        # todo construct model folder
        sess.run(tf.global_variables_initializer())

    def load(self, sess, saver):
        helper._print("Restoring model...")
        saver.restore(sess, self.model_placement)
        helper._print("Model restored!")

    def save(self, sess, saver):
        helper._print("Saving model...")
        saver.save(sess, self.model_placement)
        helper._print("Model saved!")

    def build_feed_dict(self, roots):
        roots_size = [tree_util.size_of_tree(root) for root in roots]
        roots = helper.sort_by(roots, roots_size)
        roots_size = [tree_util.size_of_tree(root) for root in roots]
        roots_list = helper.greedy_bin_packing(roots, roots_size, np.max(roots_size))

        node_list_list = []
        node_to_index_list = []
        root_indices = []
        for i, roots in enumerate(roots_list):
            node_list = []
            root_index = 0
            for root in roots:
                tree_util.depth_first_traverse(root, node_list, lambda node, node_list: node_list.append(node))
                root_index += tree_util.size_of_tree(root)
                root_indices.append([i, root_index])
            node_list_list.append(node_list)
            node_to_index = helper.reverse_dict(node_list)
            node_to_index_list.append(node_to_index)

        feed_dict = {
            # self.real_batch_size: len(node_list_list),
            self.root_array: root_indices,
            self.is_leaf_array: helper.lists_pad([
                [0] + helper.to_int([node.is_leaf for node in node_list])
                for node_list in node_list_list], 0),
            self.word_index_array: helper.lists_pad([
                [0] + [self.word_embed.get_idx(node.value) for node in node_list]
                for node_list in node_list_list], self.word_embed.get_idx("ZERO")),
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

        return feed_dict

    def construct_dir(self):
        if not os.path.exists(directories.TRAINED_MODELS_DIR):
            os.mkdir(directories.TRAINED_MODELS_DIR)
        if not os.path.exists(directories.TRAINED_MODELS_DIR + FLAGS.model_name):
            os.mkdir(directories.TRAINED_MODELS_DIR + FLAGS.model_name)

    def get_initializers(self):
        xavier_initializer = tf.contrib.layers.xavier_initializer()
        bias_initializer = tf.initializers.zeros()

        def custom_initializer(shape_list, dtype, partition_info):
            return tf.initializers.identity(gain=0.5)(shape_list, dtype,
                                                      partition_info) + tf.initializers.random_uniform(minval=-0.05,
                                                                                                       maxval=0.05)(
                shape_list, dtype, partition_info)

        weight_initializer = custom_initializer
        return xavier_initializer, weight_initializer, bias_initializer

    def predict(self, data, sess):
        helper._print_subheader("Predicting")
        feed_dict = self.build_feed_dict(data)
        return sess.run(self.p, feed_dict=feed_dict)

    def predict_and_label(self, data, sess):
        helper._print_subheader("Predicting")
        feed_dict = self.build_feed_dict(data)
        return sess.run([self.p, self.labels], feed_dict=feed_dict)


    def accuracy(self, data, sess):
        helper._print_subheader("Computing accuracy")
        feed_dict = self.build_feed_dict(data)
        return sess.run(self.acc, feed_dict=feed_dict)
