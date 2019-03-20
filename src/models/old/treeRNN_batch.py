import tensorflow as tf

from utils import constants
from utils.flags import FLAGS
import utils.tree_util as tree_util
import utils.helper as helper
from models.trees.treeModel import treeModel
import numpy as np

class treeRNN_GPU(treeModel):

    def build_constants(self):
        # embedding
        self.embeddings = tf.constant(self.data.word_embed_util.embeddings)
        ## dummi values
        self.rep_zero = tf.constant(0., shape=[FLAGS.sentence_embedding_size, FLAGS.batch_size])
        self.word_zero = tf.constant(0., shape=[FLAGS.word_embedding_size, 1])
        self.label_zero = tf.constant(0., shape=[FLAGS.label_size, FLAGS.batch_size])

    def build_placeholders(self):
        # tree structure placeholders
        self.root_array = tf.placeholder(tf.int32, (None), name='root_array')
        self.is_leaf_array = tf.placeholder(tf.bool, (None, None), name='is_leaf_array')
        self.word_index_array = tf.placeholder(tf.int32, (None, None), name='word_index_array')
        self.left_child_array = tf.placeholder(tf.int32, (None, None), name='left_child_array')
        self.right_child_array = tf.placeholder(tf.int32, (None, None), name='right_child_array')
        self.label_array = tf.placeholder(tf.int32, (None, None, FLAGS.label_size), name='label_array')

    def build_variables(self):
        # initializers
        xavier_initializer = tf.contrib.layers.xavier_initializer()

        # word variables
        self.W = tf.get_variable(name='W',  # shape=[sentence_embedding_size, word_embedding_size],
                                 initializer=tf.constant(1., shape=[FLAGS.sentence_embedding_size,
                                                                    FLAGS.word_embedding_size]))

        # phrase weights
        self.U_L = tf.get_variable(name='U_L', shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                   initializer=xavier_initializer)
        self.U_R = tf.get_variable(name='U_R', shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                   initializer=xavier_initializer)

        # bias
        self.b = tf.get_variable(name='b', initializer=tf.constant(100., shape=[FLAGS.sentence_embedding_size, 1]))

        # classifier weights
        self.V = tf.get_variable(name='V', shape=[FLAGS.label_size, FLAGS.sentence_embedding_size],
                                 initializer=xavier_initializer)
        self.b_p = tf.get_variable(name='b_p', shape=[FLAGS.label_size, 1], initializer=xavier_initializer)

    def build_model(self):
        rep_array = tf.TensorArray(
            tf.float32,
            size=0,
            dynamic_size=True,
            clear_after_read=False,
            infer_shape=False)
        rep_array = rep_array.write(0, self.rep_zero)

        word_array = tf.TensorArray(
            tf.float32,
            size=0,
            dynamic_size=True,
            clear_after_read=False,
            infer_shape=False)
        word_array = word_array.write(0, self.word_zero)

        o_array = tf.TensorArray(
            tf.float32,
            size=0,
            dynamic_size=True,
            clear_after_read=False,
            infer_shape=False)
        o_array = o_array.write(0, self.label_zero)

        def embed_word(word_index):
            return tf.transpose(tf.nn.embedding_lookup(self.embeddings, word_index))

        # todo check transpose perm
        # batch_indices = [[[j, i, j] for j in range(FLAGS.batch_size)] for i in range(FLAGS.sentence_embedding_size)]
        #
        # def gather_rep(step, children_indices, rep_array):
        #     children = tf.squeeze(tf.gather(children_indices, step, axis=1))
        #     return tf.gather_nd(rep_array.gather(children), batch_indices)

        batch_indices = [[j, j] for j in range(FLAGS.batch_size)]

        def gather_rep(step, children_indices, rep_array):
            children = tf.squeeze(tf.gather(children_indices, step, axis=1))
            rep_entries = rep_array.gather(children)
            t_rep_entries = tf.transpose(rep_entries, perm=[0, 2, 1])
            batch_size = tf.size(children)
            batch_indices = tf.stack([tf.range(batch_size), tf.range(batch_size)], axis=1)
            return tf.transpose(tf.gather_nd(t_rep_entries, batch_indices))

        def build_node(i, rep_array, word_array):
            rep_l = gather_rep(i, self.left_child_array, rep_array)
            rep_r = gather_rep(i, self.right_child_array, rep_array)
            rep_word = word_array.read(i)

            left = tf.matmul(self.U_L, rep_l)
            right = tf.matmul(self.U_R, rep_r)
            word = tf.matmul(self.W, rep_word)

            return tf.nn.leaky_relu(word + left + right + self.b)

        def tree_construction_body(rep_array, word_array, o_array, i):
            word_index = tf.gather(self.word_index_array, i, axis=1)
            word_emb = embed_word(word_index)
            word_array = word_array.write(i, word_emb)

            rep = build_node(i, rep_array, word_array)
            rep_array = rep_array.write(i, rep)

            o = tf.matmul(self.V, rep) + self.b_p
            o_array = o_array.write(i, o)

            i = tf.add(i, 1)
            return rep_array, word_array, o_array, i

        termination_cond = lambda rep_a, word_a, o_a, i: tf.less(i, tf.gather(tf.shape(self.is_leaf_array), 1))

        self.rep_array, self.word_array, self.o_array, _ = tf.while_loop(
            cond=termination_cond,
            body=tree_construction_body,
            loop_vars=(rep_array, word_array, o_array, 1),
            parallel_iterations=1
        )

    def build_loss(self):
        # todo fix loss - might drag the bias to zero??? tests and fix it
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.reshape(self.o_array.stack(), [-1, FLAGS.label_size]),
                                                       # stacking o_array this way might be wrong
                                                       labels=self.label_array))

    def build_accuracy(self):
        #roots_pad = tf.constant([i for i in range(FLAGS.batch_size)])
        #roots_padded = tf.stack([roots_pad, self.root_array], axis=1)

        logists = tf.gather_nd(tf.transpose(self.o_array.stack(), perm=[2, 0, 1]), self.root_array)#roots_padded)
        labels = tf.gather_nd(self.label_array, self.root_array)#roots_padded)

        logists_max = tf.argmax(logists, axis=1)
        labels_max = tf.argmax(labels, axis=1)

        acc = tf.equal(logists_max, labels_max)
        self.acc = tf.reduce_mean(tf.cast(acc, tf.float32))

    def build_train_op(self):
        self.global_step = tf.train.create_global_step()

        if FLAGS.lr_decay:
            n = int(len(self.data.train_trees) / FLAGS.batch_size)
            total_steps = FLAGS.epochs * n
            decay_steps = n
            decay_rate = (FLAGS.learning_rate_end / FLAGS.learning_rate) ** (decay_steps / total_steps)
            self.learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, self.global_step, decay_steps,
                                                            decay_rate,
                                                            name='learning_rate')

            helper._print_header("Using learning rate with exponential decay")
            helper._print("Decay for every step:", decay_rate)
            helper._print("Learning rate start:", FLAGS.learning_rate)
            helper._print("Learning rate end:", FLAGS.learning_rate_end)
            helper._print("After number of epochs", FLAGS.epochs)
        else:
            self.learning_rate = tf.constant(FLAGS.learning_rate)

        if FLAGS.optimizer == constants.ADAM_OPTIMIZER:
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)
        else:  # FLAGS.optimizer == constants.ADAGRAD_OPTIMIZER:
            self.train_op = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss,
                                                                                   global_step=self.global_step)

    def build_feed_dict_old(self, roots):
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
                for node_list in node_list_list], self.data.word_embed_util.get_idx("ZERO")),
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

    # def build_feed_dict(self, roots):
    #     roots_size = [tree_util.size_of_tree(root) for root in roots]
    #     roots = helper.sort_by(roots, roots_size)
    #     roots_size = [tree_util.size_of_tree(root) for root in roots]
    #     roots_list = helper.greedy_bin_packing(roots, roots_size, np.max(roots_size))
    #
    #     node_list_list = []
    #     node_to_index_list = []
    #     root_indices = []
    #     for i, roots in enumerate(roots_list):
    #         node_list = []
    #         root_index = 0
    #         for root in roots:
    #             tree_util.depth_first_traverse(root, node_list, lambda node, node_list: node_list.append(node))
    #             root_index += tree_util.size_of_tree(root)
    #             root_indices.append([i,root_index])
    #         node_list_list.append(node_list)
    #         node_to_index = helper.reverse_dict(node_list)
    #         node_to_index_list.append(node_to_index)
    #
    #     feed_dict = {
    #         self.root_array: root_indices,
    #         self.is_leaf_array: helper.lists_pad([
    #             [False] + [node.is_leaf for node in node_list]
    #             for node_list in node_list_list], False),
    #         self.word_index_array: helper.lists_pad([
    #             [0] + [self.data.word_embed_util.get_idx(node.value) for node in node_list]
    #             for node_list in node_list_list], self.data.word_embed_util.get_idx("ZERO")),
    #         self.left_child_array: helper.lists_pad([
    #             [0] + helper.add_one(
    #                 [node_to_index[node.left_child] if node.left_child is not None else -1 for node in node_list])
    #             for node_list, node_to_index in zip(node_list_list, node_to_index_list)], 0),
    #         self.right_child_array: helper.lists_pad([
    #             [0] + helper.add_one(
    #                 [node_to_index[node.right_child] if node.right_child is not None else -1 for node in node_list])
    #             for node_list, node_to_index in zip(node_list_list, node_to_index_list)], 0),
    #         self.label_array: helper.lists_pad([
    #             [[0, 0]] + [node.label for node in node_list]
    #             for node_list in node_list_list], [0, 0])
    #     }
    #
    #     return feed_dict
