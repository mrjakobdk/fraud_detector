import sys
import tensorflow as tf
import utils.helper as helper
from utils.flags import FLAGS
import os
from utils import tree_util, directories, constants
import numpy as np


class treeModel:
    def __init__(self, data, word_embed, model_name=FLAGS.model_name,
                 label_size=FLAGS.label_size,
                 learning_rate=FLAGS.learning_rate,
                 learning_rate_end=FLAGS.learning_rate_end,
                 lr_decay=FLAGS.lr_decay, use_root_loss=FLAGS.use_root_loss,
                 batch_size=FLAGS.batch_size, optimizer=FLAGS.optimizer, act_fun=FLAGS.act_fun):

        # config
        self.data = data
        self.word_embed = word_embed
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.learning_rate_end = learning_rate_end
        self.label_size = label_size
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.use_root_loss = use_root_loss

        # select activation function
        activation_functions = {"relu": tf.nn.relu, "leaky_relu": tf.nn.leaky_relu, "softplus": tf.nn.softplus,
                                "sigmoid": tf.nn.sigmoid, "tanh": tf.nn.tanh}
        self.activation_function = activation_functions[act_fun]

        self.build_constants()
        self.build_placeholders()
        self.build_variables()
        self.build_model()
        self.build_rep()
        # print_op = tf.print("reps:", self.sentence_representations, self.U_L, self.U_R, self.b_U,
        #                     output_stream=sys.stdout)
        # with tf.control_dependencies([print_op]):
        self.build_loss()
        self.build_regularization()
        self.build_accuracy()
        self.build_predict()
        self.build_train_op()

    def build_constants(self):
        raise NotImplementedError("Each Model must re-implement this method.")

    def build_placeholders(self):
        # tree structure placeholders
        self.dropout_rate = tf.placeholder(tf.float32, None, name='dropout_rate')
        self.loss_array = tf.placeholder(tf.int32, (None, None), name='loss_array')
        self.root_array = tf.placeholder(tf.int32, (None, None), name='root_array')
        self.is_leaf_array = tf.placeholder(tf.float32, (None, None), name='is_leaf_array')
        self.word_index_array = tf.placeholder(tf.int32, (None, None), name='word_index_array')
        self.left_child_array = tf.placeholder(tf.int32, (None, None), name='left_child_array')
        self.right_child_array = tf.placeholder(tf.int32, (None, None), name='right_child_array')
        self.label_array = tf.placeholder(tf.int32, (None, None, FLAGS.label_size), name='label_array')
        self.real_batch_size = tf.gather(tf.shape(self.is_leaf_array), 0)

    def build_variables(self):
        raise NotImplementedError("Each Model must re-implement this method.")

    def build_model(self):
        raise NotImplementedError("Each Model must re-implement this method.")

    def build_loss(self):
        logits = tf.gather_nd(tf.transpose(self.o_array.stack(), perm=[2, 0, 1]), self.loss_array)
        labels = tf.gather_nd(self.label_array, self.loss_array)

        softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
        self.loss = tf.reduce_mean(softmax_cross_entropy)

        # self.loss = tf.reduce_mean(
        #     tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.reshape(self.o_array.stack(), [-1, FLAGS.label_size]),
        #                                                # stacking o_array this way might be wrong
        #                                                labels=self.label_array))

        # reg_weight = 0.001
        # self.loss += reg_weight * tf.nn.l2_loss(self.W)
        # self.loss += reg_weight * tf.nn.l2_loss(self.U_L)
        # self.loss += reg_weight * tf.nn.l2_loss(self.U_R)
        # self.loss += reg_weight * tf.nn.l2_loss(self.V)

    def build_regularization(self):
        for weight in self.reg_weights:
            self.loss += FLAGS.l2_scalar * tf.nn.l2_loss(weight)


    def build_rep(self):
        self.sentence_representations = tf.gather_nd(tf.transpose(self.rep_array.stack(), perm=[2, 0, 1]),
                                                     self.root_array)

    def build_accuracy(self):
        logits = tf.gather_nd(tf.transpose(self.o_array.stack(), perm=[2, 0, 1]), self.root_array)
        labels = tf.gather_nd(self.label_array, self.root_array)

        logits_max = tf.argmax(logits, axis=1)
        labels_max = tf.argmax(labels, axis=1)

        acc = tf.equal(logits_max, labels_max)
        self.acc = tf.reduce_mean(tf.cast(acc, tf.float32))

    def build_predict(self):
        logits = tf.gather_nd(tf.transpose(self.o_array.stack(), perm=[2, 0, 1]), self.root_array)
        self.labels = tf.gather_nd(self.label_array, self.root_array)
        self.p = tf.nn.softmax(logits, axis=-1)

    def build_train_op(self):
        self.global_step = tf.train.create_global_step()

        if self.lr_decay > 0:
            decay_steps = int(len(self.data.train_trees) / self.batch_size)
            self.lr = tf.train.exponential_decay(self.learning_rate, self.global_step, decay_steps,
                                                 self.lr_decay, name='learning_rate')

        else:
            self.lr = tf.constant(self.learning_rate)

        if self.optimizer == constants.ADAM_OPTIMIZER:
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)
        else:
            self.train_op = tf.train.AdagradOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)

    def initialize(self, sess):
        # todo construct model folder
        sess.run(tf.global_variables_initializer())

    def build_feed_dict(self, roots, sort=True, train=False):
        if sort:
            roots_size = [tree_util.size_of_tree(root) for root in roots]
            roots = helper.sort_by(roots, roots_size)
        roots_size = [tree_util.size_of_tree(root) for root in roots]
        roots_list, permutation = helper.greedy_bin_packing(roots, roots_size, np.max(roots_size))

        node_list_list = []
        node_to_index_list = []
        root_indices = []
        internal_nodes_array = []
        length_of_sentence = []
        for i, roots in enumerate(roots_list):
            node_list = []
            root_index = 0
            for root in roots:
                tree_util.depth_first_traverse(root, node_list, lambda node, node_list: node_list.append(node))
                size = tree_util.size_of_tree(root)
                root_index += size
                length_of_sentence.append(size)
                root_indices.append([i, root_index])
            node_list_list.append(node_list)
            node_to_index = helper.reverse_dict(node_list)
            node_to_index_list.append(node_to_index)
            for node in node_list:
                if not node.is_leaf or FLAGS.use_leaf_loss:
                    internal_nodes_array.append([i, node_to_index[node] + 1])

        internal_nodes_array = internal_nodes_array if len(internal_nodes_array)>0 else [[0, 0]] #hack to fix case where all length a 1

        feed_dict = {
            self.dropout_rate: FLAGS.dropout_prob if train else 0,
            self.root_array: root_indices,
            self.loss_array: root_indices if self.use_root_loss else internal_nodes_array,
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

        return feed_dict, permutation

    # def construct_dir(self):
    #     if not os.path.exists(directories.TRAINED_MODELS_DIR):
    #         os.mkdir(directories.TRAINED_MODELS_DIR)
    #     if not os.path.exists(directories.TRAINED_MODELS_DIR + FLAGS.model_name):
    #         os.mkdir(directories.TRAINED_MODELS_DIR + FLAGS.model_name)

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
        feed_dict, _ = self.build_feed_dict(data)
        return sess.run(self.p, feed_dict=feed_dict)

    def predict_and_label(self, data, sess):
        helper._print_subheader("Predicting")
        prob, labels = [], []
        batches = helper.batches(data, batch_size=1000 if FLAGS.use_gpu else 2, use_tail=True, perm=False)
        for batch in batches:
            feed_dict, _ = self.build_feed_dict(batch)
            p, l = sess.run([self.p, self.labels], feed_dict=feed_dict)
            prob.extend(p)
            labels.extend(l)
        return prob, labels

    def accuracy(self, data, sess):
        helper._print_subheader("Computing accuracy")
        feed_dict, _ = self.build_feed_dict(data)
        print("roots", len(feed_dict[self.root_array]))
        return sess.run(self.acc, feed_dict=feed_dict)


    def load_best(self, sess, saver, data_set):
        helper._print(f"Restoring best {data_set} model...")
        saver.restore(sess, directories.BEST_MODEL_FILE(self.model_name, data_set))
        helper._print(f"Model best {data_set} restored!")

    def load_tmp(self, sess, saver):
        helper._print("Restoring tmp model...")
        saver.restore(sess, directories.TMP_MODEL_FILE(self.model_name))
        helper._print("Model tmp restored!")

    def save_best(self, sess, saver, data_set):
        helper._print(f"Saving best {data_set} model...")
        saver.save(sess, directories.BEST_MODEL_FILE(self.model_name, data_set))
        helper._print(f"Model best {data_set} saved!")

    def save_tmp(self, sess, saver):
        helper._print("Saving tmp model...")
        saver.save(sess, directories.TMP_MODEL_FILE(self.model_name))
        helper._print("Model tmp saved!")

    def save_pre_end(self, sess, saver, data_set):
        helper._print(f"Saving pre {data_set} model...")
        saver.save(sess, directories.PRE_MODEL_FILE(self.model_name, data_set))
        helper._print(f"Model pre {data_set} saved!")




    def get_no_trainable_variables(self):
        # https://stackoverflow.com/questions/38160940/how-to-count-total-number-of-trainable-parameters-in-a-tensorflow-model
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        return total_parameters
