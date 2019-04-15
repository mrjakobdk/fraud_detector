import tensorflow as tf
from utils.flags import FLAGS
import utils.tree_util as tree_util
import utils.helper as helper
from models.trees.treeModel import treeModel
import numpy as np
import sys


class treeLSTM_tracker(treeModel):

    def build_constants(self):
        # embedding
        self.embeddings = tf.constant(self.word_embed.embeddings)
        ## dummi values
        self.rep_zero = tf.constant(0., shape=[FLAGS.sentence_embedding_size])
        self.word_zero = tf.constant(0., shape=[FLAGS.word_embedding_size])  # todo changed from 1d
        self.label_zero = tf.constant(0., shape=[FLAGS.label_size])

    def build_placeholders(self):
        # tracker
        self.leaf_word_array = tf.placeholder(tf.int32, (None, None), name='word_index_sentence')
        self.lstm_index_array = tf.placeholder(tf.int32, (None, None), name='preceding_lstm_index')
        self.lstm_prev_array = tf.placeholder(tf.int32, (None, None), name='lstm_prev')

        # tree structure placeholders
        self.loss_array = tf.placeholder(tf.int32, (None, None), name='loss_array')
        self.root_array = tf.placeholder(tf.int32, (None, None), name='root_array')
        self.is_leaf_array = tf.placeholder(tf.bool, (None, None), name='is_leaf_array')
        self.word_index_array = tf.placeholder(tf.int32, (None, None), name='word_index_array')
        self.left_child_array = tf.placeholder(tf.int32, (None, None), name='left_child_array')
        self.right_child_array = tf.placeholder(tf.int32, (None, None), name='right_child_array')
        self.label_array = tf.placeholder(tf.int32, (None, None, FLAGS.label_size), name='label_array')
        self.real_batch_size = tf.gather(tf.shape(self.is_leaf_array), 0)

    def build_variables(self):
        # initializers
        xavier_initializer, weight_initializer, bias_initializer = self.get_initializers()

        # lstm tracker variables
        self.Wc_tracker = tf.get_variable(name='Wc_tracker',
                                          shape=[FLAGS.sentence_embedding_size, FLAGS.word_embedding_size],
                                          initializer=xavier_initializer)
        self.Wi_tracker = tf.get_variable(name='Wi_tracker',
                                          shape=[FLAGS.sentence_embedding_size, FLAGS.word_embedding_size],
                                          initializer=xavier_initializer)
        self.Wf_tracker = tf.get_variable(name='Wf_tracker',
                                          shape=[FLAGS.sentence_embedding_size, FLAGS.word_embedding_size],
                                          initializer=xavier_initializer)
        self.Wo_tracker = tf.get_variable(name='Wo_tracker',
                                          shape=[FLAGS.sentence_embedding_size, FLAGS.word_embedding_size],
                                          initializer=xavier_initializer)

        self.Uc_tracker = tf.get_variable(name='Uc_tracker',
                                          shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                          initializer=xavier_initializer)
        self.Ui_tracker = tf.get_variable(name='Ui_tracker',
                                          shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                          initializer=xavier_initializer)
        self.Uf_tracker = tf.get_variable(name='Uf_tracker',
                                          shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                          initializer=xavier_initializer)
        self.Uo_tracker = tf.get_variable(name='Uo_tracker',
                                          shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                          initializer=xavier_initializer)

        self.bc_tracker = tf.get_variable(name='bc_tracker', shape=[FLAGS.sentence_embedding_size, 1],
                                          initializer=xavier_initializer)
        self.bi_tracker = tf.get_variable(name='bi_tracker', shape=[FLAGS.sentence_embedding_size, 1],
                                          initializer=xavier_initializer)
        self.bf_tracker = tf.get_variable(name='bf_tracker', shape=[FLAGS.sentence_embedding_size, 1],
                                          initializer=xavier_initializer)
        self.bo_tracker = tf.get_variable(name='bo_tracker', shape=[FLAGS.sentence_embedding_size, 1],
                                          initializer=xavier_initializer)

        # tracker encoding variable
        self.Ec = tf.get_variable(name='Ec', shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                  initializer=xavier_initializer)
        self.Ei = tf.get_variable(name='Ei', shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                  initializer=xavier_initializer)
        self.Ef_l = tf.get_variable(name='Ef_l', shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                    initializer=xavier_initializer)
        self.Ef_r = tf.get_variable(name='Ef_r', shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                    initializer=xavier_initializer)
        self.Eo = tf.get_variable(name='Eo', shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                  initializer=xavier_initializer)

        # TreeLSTM
        self.Wi = tf.get_variable(name='Wi', shape=[FLAGS.sentence_embedding_size, FLAGS.word_embedding_size],
                                  initializer=xavier_initializer)
        self.Wf = tf.get_variable(name='Wf', shape=[FLAGS.sentence_embedding_size, FLAGS.word_embedding_size],
                                  initializer=xavier_initializer)
        self.Wo = tf.get_variable(name='Wo', shape=[FLAGS.sentence_embedding_size, FLAGS.word_embedding_size],
                                  initializer=xavier_initializer)
        self.Wc = tf.get_variable(name='Wc', shape=[FLAGS.sentence_embedding_size, FLAGS.word_embedding_size],
                                  initializer=xavier_initializer)

        # phrase weights
        self.Ui_L = tf.get_variable(name='Ui_L', shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                    initializer=xavier_initializer)
        self.Ui_R = tf.get_variable(name='Ui_R', shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                    initializer=xavier_initializer)
        self.Uf_L = tf.get_variable(name='Uf_L', shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                    initializer=xavier_initializer)
        self.Uf_R = tf.get_variable(name='Uf_R', shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                    initializer=xavier_initializer)
        self.Uo_L = tf.get_variable(name='Uo_L', shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                    initializer=xavier_initializer)
        self.Uo_R = tf.get_variable(name='Uo_R', shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                    initializer=xavier_initializer)
        self.Uc_L = tf.get_variable(name='Uc_L', shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                    initializer=xavier_initializer)
        self.Uc_R = tf.get_variable(name='Uc_R', shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                    initializer=xavier_initializer)

        # bias
        self.bi = tf.get_variable(name='bi', shape=[FLAGS.sentence_embedding_size, 1], initializer=xavier_initializer)
        self.bf = tf.get_variable(name='bf', shape=[FLAGS.sentence_embedding_size, 1], initializer=xavier_initializer)
        self.bo = tf.get_variable(name='bo', shape=[FLAGS.sentence_embedding_size, 1], initializer=xavier_initializer)
        self.bc = tf.get_variable(name='bc', shape=[FLAGS.sentence_embedding_size, 1], initializer=xavier_initializer)

        # classifier weights
        self.V = tf.get_variable(name='V', shape=[FLAGS.label_size, FLAGS.sentence_embedding_size],
                                 initializer=xavier_initializer)
        self.b_p = tf.get_variable(name='b_p', shape=[FLAGS.label_size, 1], initializer=xavier_initializer)

    def build_model(self):
        mem_array = tf.TensorArray(
            tf.float32,
            size=0,
            dynamic_size=True,
            clear_after_read=False,
            infer_shape=False)
        mem_array = mem_array.write(0, tf.reshape(tf.tile(self.rep_zero, tf.stack([self.real_batch_size])),
                                                  [FLAGS.sentence_embedding_size, -1]))

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

        rep_array = tf.TensorArray(
            tf.float32,
            size=0,
            dynamic_size=True,
            clear_after_read=False,
            infer_shape=False)
        rep_array = rep_array.write(0, tf.reshape(tf.tile(self.rep_zero, tf.stack([self.real_batch_size])),
                                                  [FLAGS.sentence_embedding_size, -1]))

        word_array = tf.TensorArray(
            tf.float32,
            size=0,
            dynamic_size=True,
            clear_after_read=False,
            infer_shape=False)
        word_array = word_array.write(0, self.word_zero)  # todo this might not be right

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
            # t_prev = tf.gather(self.lstm_prev_array, t, axis=1)
            e_prev = gather_rep(t, self.lstm_prev_array, e_array)
            c_prev = gather_rep(t, self.lstm_prev_array, c_array)
            # e_prev = e_array.read(t_prev)
            # c_prev = c_array.read(t_prev)
            w_t = w_array.read(t)

            u_t = tf.tanh(tf.matmul(self.Wc_tracker, w_t) + tf.matmul(self.Uc_tracker, e_prev) + self.bc_tracker)
            i_t = tf.sigmoid(tf.matmul(self.Wi_tracker, w_t) + tf.matmul(self.Ui_tracker, e_prev) + self.bi_tracker)
            f_t = tf.sigmoid(tf.matmul(self.Wf_tracker, w_t) + tf.matmul(self.Uf_tracker, e_prev) + self.bf_tracker)
            o_t = tf.sigmoid(tf.matmul(self.Wo_tracker, w_t) + tf.matmul(self.Uo_tracker, e_prev) + self.bo_tracker)

            c_t = tf.math.multiply(i_t, u_t) + tf.math.multiply(f_t, c_prev)

            e_t = tf.math.multiply(o_t, tf.tanh(c_t))
            return e_t, c_t

        def lstm_construction_body(t, e_array, c_array, w_array):
            word_index = tf.gather(self.leaf_word_array, t, axis=1)
            word_emb = embed_word(word_index)
            w_array = w_array.write(t, word_emb)

            e, c = build_lstm_cell(t, e_array, c_array, w_array)
            e_array = e_array.write(t, e)
            c_array = c_array.write(t, c)

            t = tf.add(t, 1)
            return t, e_array, c_array, w_array

        termination_cond = lambda t, e_array, c_array, w_array: tf.less(t, tf.gather(tf.shape(self.leaf_word_array), 1))
        _, self.e_array, self.c_array, self.w_array = tf.while_loop(
            cond=termination_cond,
            body=lstm_construction_body,
            loop_vars=(1, e_array, c_array, w_array),
            parallel_iterations=1
        )

        def build_node(i, rep_array, word_array, mem_array, e_array):
            rep_l = gather_rep(i, self.left_child_array, rep_array)
            rep_r = gather_rep(i, self.right_child_array, rep_array)
            c_l = gather_rep(i, self.left_child_array, mem_array)
            c_r = gather_rep(i, self.right_child_array, mem_array)
            rep_word = word_array.read(i)
            e = gather_rep(i, self.lstm_index_array, e_array)

            tracker_i = tf.matmul(self.Ei, e)
            tracker_f_l = tf.matmul(self.Ef_l, e)
            tracker_f_r = tf.matmul(self.Ef_r, e)
            tracker_o = tf.matmul(self.Eo, e)
            tracker_c = tf.matmul(self.Ec, e)

            i_n = tf.sigmoid(
                tf.matmul(self.Wi, rep_word) + tf.matmul(self.Ui_L, rep_l) + tf.matmul(self.Ui_R,
                                                                                       rep_r) + tracker_i + self.bi)

            f_l = tf.sigmoid(
                tf.matmul(self.Wf, rep_word) + tf.matmul(self.Uf_R, rep_l) + tracker_f_l + self.bf)

            f_r = tf.sigmoid(
                tf.matmul(self.Wf, rep_word) + tf.matmul(self.Uf_L, rep_r) + tracker_f_r + self.bf)

            o_n = tf.sigmoid(
                tf.matmul(self.Wo, rep_word) + tf.matmul(self.Uo_L, rep_l) + tf.matmul(self.Uo_R,
                                                                                       rep_r) + tracker_o + self.bo)

            u_n = tf.tanh(
                tf.matmul(self.Wc, rep_word) + tf.matmul(self.Uc_L, rep_l) + tf.matmul(self.Uc_R,
                                                                                       rep_r) + tracker_c + self.bc)

            c_n = tf.math.multiply(i_n, u_n) + tf.math.multiply(f_r, c_l) + tf.math.multiply(f_l, c_r)
            h_n = tf.math.multiply(o_n, tf.tanh(c_n))

            return h_n, c_n

        def tree_construction_body(rep_array, word_array, o_array, mem_array, e_array, i):
            word_index = tf.gather(self.word_index_array, i, axis=1)
            # print_op = tf.print("word_index:", word_index,
            #                     output_stream=sys.stdout)
            # with tf.control_dependencies([print_op]):
            word_emb = embed_word(word_index)
            word_array = word_array.write(i, word_emb)

            rep, c = build_node(i, rep_array, word_array, mem_array, e_array)
            rep_array = rep_array.write(i, rep)
            mem_array = mem_array.write(i, c)

            o = tf.matmul(self.V, rep) + self.b_p
            o_array = o_array.write(i, o)

            i = tf.add(i, 1)
            return rep_array, word_array, o_array, mem_array, e_array, i

        termination_cond = lambda rep_a, word_a, o_a, m_a, e_a, i: tf.less(i, tf.gather(tf.shape(self.left_child_array),
                                                                                        1))

        self.rep_array, self.word_array, self.o_array, self.mem_array, self.e_array, _ = tf.while_loop(
            cond=termination_cond,
            body=tree_construction_body,
            loop_vars=(rep_array, word_array, o_array, mem_array, self.e_array, 1),
            parallel_iterations=1
        )

    def build_feed_dict(self, roots, sort=True):
        if sort:
            roots_size = [tree_util.size_of_tree(root) for root in roots]
            roots = helper.sort_by(roots, roots_size)
        roots_size = [tree_util.size_of_tree(root) for root in roots]
        roots_list, permutation = helper.greedy_bin_packing(roots, roots_size, np.max(roots_size))

        node_list_list = []
        node_to_index_list = []
        root_indices = []
        lstm_idx_list = []
        lstm_prev_list = []
        internal_nodes_array = []
        for i, roots in enumerate(roots_list):
            node_list = []
            lstm_idx = [0]
            lstm_prev = [0]
            lstm_prev_count = 0
            root_index = 0
            start = 0
            for root in roots:
                tree_util.depth_first_traverse(root, node_list, lambda node, node_list: node_list.append(node))

                _, start = tree_util.get_preceding_lstm_index(root, start, start, lstm_idx)

                leaf_count = tree_util.leafs_in_tree(root)
                for x in range(0, leaf_count):
                    if x == 0:
                        lstm_prev.append(0)
                    else:
                        lstm_prev.append(lstm_prev_count)
                    lstm_prev_count += 1

                root_index += tree_util.size_of_tree(root)
                root_indices.append([i, root_index])
            node_list_list.append(node_list)
            node_to_index = helper.reverse_dict(node_list)
            node_to_index_list.append(node_to_index)
            lstm_idx_list.append(lstm_idx)
            lstm_prev_list.append(lstm_prev)
            for node in node_list:
                if not node.is_leaf:
                    internal_nodes_array.append([i, node_to_index[node] + 1])

        internal_nodes_array = internal_nodes_array if len(internal_nodes_array) > 0 else [[0, 0]]
        # print(lstm_prev_list)
        feed_dict = {
            self.lstm_prev_array: helper.lists_pad(lstm_prev_list, 0),
            self.leaf_word_array: helper.lists_pad(
                [[0] + [self.word_embed.get_idx(node.value) for node in node_list if node.is_leaf]
                 for node_list in node_list_list]
                , 0),
            self.lstm_index_array: helper.lists_pad(
                lstm_idx_list
                , 0),
            self.loss_array: root_indices if self.use_root_loss else internal_nodes_array,
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

        return feed_dict, permutation
