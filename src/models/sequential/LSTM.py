import tensorflow as tf
import numpy as np

from models.trees.treeModel import treeModel
from utils import tree_util, helper
from utils.flags import FLAGS


class LSTM(treeModel):

    def build_constants(self):
        # embedding
        self.embeddings = tf.constant(self.word_embed.embeddings)
        ## dummi values
        self.rep_zero = tf.constant(0., shape=[FLAGS.sentence_embedding_size])
        self.word_zero = tf.constant(0., shape=[FLAGS.word_embedding_size]) #todo changed from 1d
        self.label_zero = tf.constant(0., shape=[FLAGS.label_size])

    def build_placeholders(self):
        self.lstm_prev_array = tf.placeholder(tf.int32, (None, None), name='lstm_prev')
        self.leaf_word_array = tf.placeholder(tf.int32, (None, None), name='word_index_sentence')
        self.loss_array = tf.placeholder(tf.int32, (None, None), name='loss_array')
        self.root_array = tf.placeholder(tf.int32, (None, None), name='root_array')
        self.label_array = tf.placeholder(tf.int32, (None, None, FLAGS.label_size), name='label_array')
        self.real_batch_size = tf.gather(tf.shape(self.leaf_word_array), 0)

    def build_variables(self):
        # initializers
        xavier_initializer, weight_initializer, bias_initializer = self.get_initializers()

        # lstm variables
        self.Wc = tf.get_variable(name='Wc', shape=[FLAGS.sentence_embedding_size, FLAGS.word_embedding_size],
                                 initializer=xavier_initializer)
        self.Wi = tf.get_variable(name='Wi', shape=[FLAGS.sentence_embedding_size, FLAGS.word_embedding_size],
                                  initializer=xavier_initializer)
        self.Wf = tf.get_variable(name='Wf', shape=[FLAGS.sentence_embedding_size, FLAGS.word_embedding_size],
                                  initializer=xavier_initializer)
        self.Wo = tf.get_variable(name='Wo', shape=[FLAGS.sentence_embedding_size, FLAGS.word_embedding_size],
                                  initializer=xavier_initializer)

        self.Uc = tf.get_variable(name='Uc', shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                  initializer=xavier_initializer)
        self.Ui = tf.get_variable(name='Ui', shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                  initializer=xavier_initializer)
        self.Uf = tf.get_variable(name='Uf', shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                  initializer=xavier_initializer)
        self.Uo = tf.get_variable(name='Uo', shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                  initializer=xavier_initializer)

        self.bc = tf.get_variable(name='bc', shape=[FLAGS.sentence_embedding_size, 1], initializer=xavier_initializer)
        self.bi = tf.get_variable(name='bi', shape=[FLAGS.sentence_embedding_size, 1], initializer=xavier_initializer)
        self.bf = tf.get_variable(name='bf', shape=[FLAGS.sentence_embedding_size, 1], initializer=xavier_initializer)
        self.bo = tf.get_variable(name='bo', shape=[FLAGS.sentence_embedding_size, 1], initializer=xavier_initializer)

        # classifier weights
        self.V = tf.get_variable(name='V', shape=[FLAGS.label_size, FLAGS.sentence_embedding_size],
                                 initializer=xavier_initializer)
        self.b_p = tf.get_variable(name='b_p', shape=[FLAGS.label_size, 1], initializer=xavier_initializer)

    def build_model(self):
        e_array = tf.TensorArray(
            tf.float32,
            size=0,
            dynamic_size=True,
            clear_after_read=False,
            infer_shape=False)
        e_array = e_array.write(0, tf.reshape(tf.tile(self.rep_zero, tf.stack([self.real_batch_size])),
                                              [FLAGS.sentence_embedding_size, -1]))

        c_array = tf.TensorArray(
            tf.float32,
            size=0,
            dynamic_size=True,
            clear_after_read=False,
            infer_shape=False)
        c_array = c_array.write(0, tf.reshape(tf.tile(self.rep_zero, tf.stack([self.real_batch_size])),
                                              [FLAGS.sentence_embedding_size, -1]))

        w_array = tf.TensorArray(
            tf.float32,
            size=0,
            dynamic_size=True,
            clear_after_read=False,
            infer_shape=False)
        w_array = w_array.write(0, tf.reshape(tf.tile(self.word_zero, tf.stack([self.real_batch_size])),
                                              [FLAGS.word_embedding_size, -1]))

        o_array = tf.TensorArray(
            tf.float32,
            size=0,
            dynamic_size=True,
            clear_after_read=False,
            infer_shape=False)
        o_array = o_array.write(0, tf.reshape(tf.tile(self.label_zero, tf.stack([self.real_batch_size])),
                                              [FLAGS.label_size, -1]))

        def embed_word(word_index):
            return tf.transpose(tf.nn.embedding_lookup(self.embeddings, word_index))

        batch_indices = tf.stack([tf.range(self.real_batch_size), tf.range(self.real_batch_size)], axis=1)

        def gather_rep(step, children_indices, rep_array):
            children = tf.squeeze(tf.gather(children_indices, step, axis=1))
            rep_entries = rep_array.gather(children)
            t_rep_entries = tf.transpose(rep_entries, perm=[0, 2, 1])
            return tf.transpose(tf.gather_nd(t_rep_entries, batch_indices))

        def build_lstm_cell(t, e_array, c_array, w_array):
            e_prev = gather_rep(t, self.lstm_prev_array, e_array)
            c_prev = gather_rep(t, self.lstm_prev_array, c_array)
            w_t = w_array.read(t)

            u_t = tf.tanh(tf.matmul(self.Wc, w_t) + tf.matmul(self.Uc, e_prev) + self.bc)
            i_t = tf.sigmoid(tf.matmul(self.Wi, w_t) + tf.matmul(self.Ui, e_prev) + self.bi)
            f_t = tf.sigmoid(tf.matmul(self.Wf, w_t) + tf.matmul(self.Uf, e_prev) + self.bf)
            o_t = tf.sigmoid(tf.matmul(self.Wo, w_t) + tf.matmul(self.Uo, e_prev) + self.bo)

            c_t = tf.math.multiply(i_t, u_t) + tf.math.multiply(f_t, c_prev)

            e_t = tf.math.multiply(o_t, tf.tanh(c_t))
            return e_t, c_t

        def lstm_construction_body(t, e_array, c_array, w_array, o_array):
            word_index = tf.gather(self.leaf_word_array, t, axis=1)
            word_emb = embed_word(word_index)
            w_array = w_array.write(t, word_emb)

            e, c = build_lstm_cell(t, e_array, c_array, w_array)
            e_array = e_array.write(t, e)
            c_array = c_array.write(t, c)

            o = tf.matmul(self.V, e) + self.b_p
            o_array = o_array.write(t, o)

            t = tf.add(t, 1)
            return t, e_array, c_array, w_array, o_array

        termination_cond = lambda t, e_array, c_array, w_array, o_array: tf.less(t, tf.gather(tf.shape(self.leaf_word_array), 1))
        _, self.e_array, self.c_array, self.w_array, self.o_array = tf.while_loop(
            cond=termination_cond,
            body=lstm_construction_body,
            loop_vars=(1, e_array, c_array, w_array, o_array),
            parallel_iterations=1
        )

    def build_feed_dict(self, roots, sort=True):
        if sort:
            roots_size = [tree_util.size_of_tree(root) for root in roots]
            roots = helper.sort_by(roots, roots_size)
        roots_size = [tree_util.size_of_tree(root) for root in roots]
        roots_list, permutation = helper.greedy_bin_packing(roots, roots_size, np.max(roots_size))

        node_list_list = []
        root_indices = []
        internal_nodes_array = []
        lstm_prev_list = []
        for i, roots in enumerate(roots_list):
            node_list = []
            root_index = 0
            leaf_index = 0
            lstm_prev = [0]
            lstm_prev_count = 0
            for root in roots:
                tree_util.depth_first_traverse(root, node_list, lambda node, node_list: node_list.append(node))
                leaf_count = tree_util.leafs_in_tree(root)
                root_index += leaf_count
                root_indices.append([i, root_index])
                for j in range(0,leaf_count):
                    leaf_index += 1
                    internal_nodes_array.append([i, leaf_index])

                leaf_count = tree_util.leafs_in_tree(root)
                for x in range(0, leaf_count):
                    if x == 0:
                        lstm_prev.append(0)
                    else:
                        lstm_prev.append(lstm_prev_count)
                    lstm_prev_count += 1

            node_list_list.append(node_list)
            lstm_prev_list.append(lstm_prev)

        feed_dict = {
            self.lstm_prev_array: helper.lists_pad(lstm_prev_list, 0),
            self.leaf_word_array: helper.lists_pad(
                [[0] + [self.word_embed.get_idx(node.value) for node in node_list if node.is_leaf]
                for node_list in node_list_list]
            ,0),
            self.loss_array: root_indices if self.use_root_loss else internal_nodes_array,
            self.root_array: root_indices,
            self.label_array: helper.lists_pad([
                [[0, 0]] + [node.label for node in node_list if node.is_leaf]
                for node_list in node_list_list], [0, 0])
        }

        return feed_dict, permutation

    def build_rep(self):
        self.sentence_representations = tf.gather_nd(tf.transpose(self.e_array.stack(), perm=[2, 0, 1]),
                                                     self.root_array)

    # def build_loss(self):
    #     logits = tf.gather_nd(tf.transpose(self.o_array.stack(), perm=[2, 0, 1]), self.loss_array)
    #     labels = tf.gather_nd(self.label_array, self.loss_array)
    #
    #     softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
    #     self.loss = tf.reduce_mean(softmax_cross_entropy)
    #
    # def build_accuracy(self):
    #     logits = tf.gather_nd(tf.transpose(self.o_array.stack(), perm=[2, 0, 1]), self.root_array)
    #     labels = tf.gather_nd(self.label_array, self.root_array)
    #
    #     logits_max = tf.argmax(logits, axis=1)
    #     labels_max = tf.argmax(labels, axis=1)
    #
    #     acc = tf.equal(logits_max, labels_max)
    #     self.acc = tf.reduce_mean(tf.cast(acc, tf.float32))