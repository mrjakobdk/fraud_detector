import tensorflow as tf
from utils.flags import FLAGS
from models.trees.treeModel import treeModel
import sys


class treeRNN_neerbek(treeModel):

    def build_constants(self):
        # embedding
        self.embeddings = tf.constant(self.word_embed.embeddings)
        ## dummi values
        self.rep_zero = tf.constant(0., shape=[FLAGS.sentence_embedding_size])
        self.word_zero = tf.constant(0., shape=[FLAGS.word_embedding_size])
        self.label_zero = tf.constant(0., shape=[FLAGS.label_size])

    # def build_placeholders(self):
    #     # tree structure placeholders
    #     self.loss_array = tf.placeholder(tf.int32, (None, None), name='loss_array')
    #     self.root_array = tf.placeholder(tf.int32, (None, None), name='root_array')
    #     self.is_leaf_array = tf.placeholder(tf.float32, (None, None), name='is_leaf_array')
    #     self.word_index_array = tf.placeholder(tf.int32, (None, None), name='word_index_array')
    #     self.left_child_array = tf.placeholder(tf.int32, (None, None), name='left_child_array')
    #     self.right_child_array = tf.placeholder(tf.int32, (None, None), name='right_child_array')
    #     self.label_array = tf.placeholder(tf.int32, (None, None, FLAGS.label_size), name='label_array')
    #     self.real_batch_size = tf.gather(tf.shape(self.is_leaf_array), 0)

    def build_variables(self):
        # initializers
        xavier_initializer, weight_initializer, bias_initializer = self.get_initializers()

        # word variables
        self.W_L = tf.get_variable(name='W_L', shape=[FLAGS.sentence_embedding_size, FLAGS.word_embedding_size],
                                   initializer=xavier_initializer)
        self.W_R = tf.get_variable(name='W_R', shape=[FLAGS.sentence_embedding_size, FLAGS.word_embedding_size],
                                   initializer=xavier_initializer)

        # phrase weights
        self.U_L = tf.get_variable(name='U_L', shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                   initializer=xavier_initializer)
        self.U_R = tf.get_variable(name='U_R', shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                   initializer=xavier_initializer)

        # bias
        self.b_W = tf.get_variable(name='b_W', shape=[FLAGS.sentence_embedding_size, 1], initializer=xavier_initializer)
        self.b_U = tf.get_variable(name='b_U', shape=[FLAGS.sentence_embedding_size, 1], initializer=xavier_initializer)

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
        rep_array = rep_array.write(0, tf.reshape(tf.tile(self.rep_zero, tf.stack([self.real_batch_size])),
                                                  [FLAGS.sentence_embedding_size, -1]))

        word_array = tf.TensorArray(
            tf.float32,
            size=0,
            dynamic_size=True,
            clear_after_read=False,
            infer_shape=False)
        word_array = word_array.write(0, tf.reshape(tf.tile(self.word_zero, tf.stack([self.real_batch_size])),
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

        def build_node(i, rep_array, word_array):
            left_child = tf.stack([tf.range(self.real_batch_size), tf.gather(self.left_child_array, i, axis=1)], axis=1)
            right_child = tf.stack([tf.range(self.real_batch_size), tf.gather(self.right_child_array, i, axis=1)],
                                   axis=1)

            is_leaf = tf.reshape(tf.gather(self.is_leaf_array, i, axis=1), shape=(1, -1))
            is_leaf_l = tf.reshape(tf.gather_nd(self.is_leaf_array, left_child), shape=(1, -1))
            is_leaf_r = tf.reshape(tf.gather_nd(self.is_leaf_array, right_child), shape=(1, -1))

            rep_l = gather_rep(i, self.left_child_array, rep_array)
            rep_r = gather_rep(i, self.right_child_array, rep_array)
            rep_word_l = gather_rep(i, self.left_child_array, word_array)
            rep_word_r = gather_rep(i, self.right_child_array, word_array)

            phrase_left = tf.matmul(self.U_L, rep_l) + tf.matmul(self.b_U, 1. - is_leaf_l)  # 1 - to negate is_leaf
            phrase_right = tf.matmul(self.U_R, rep_r) + tf.matmul(self.b_U, 1. - is_leaf_r)
            word_left = tf.matmul(self.W_L, rep_word_l) + tf.matmul(self.b_W, is_leaf_l)
            word_right = tf.matmul(self.W_R, rep_word_r) + tf.matmul(self.b_W, is_leaf_r)

            is_node = 1. - tf.squeeze(is_leaf)
            is_node_diag = tf.linalg.tensor_diag(is_node)

            return tf.matmul(tf.nn.relu(word_left + phrase_left + word_right + phrase_right), is_node_diag)

        def tree_construction_body(rep_array, word_array, o_array, i):
            word_index = tf.gather(self.word_index_array, i, axis=1)
            # print_op = tf.print("word_index:", word_index,
            #                     output_stream=sys.stdout)
            # with tf.control_dependencies([print_op]):
            word_emb = embed_word(word_index)
            word_array = word_array.write(i, word_emb)

            rep = build_node(i, rep_array, word_array)
            rep_array = rep_array.write(i, rep)

            o = tf.matmul(self.V, rep) + self.b_p
            o_array = o_array.write(i, o)

            i = tf.add(i, 1)
            return rep_array, word_array, o_array, i

        termination_cond = lambda rep_a, word_a, o_a, i: tf.less(i, tf.gather(tf.shape(self.left_child_array), 1))

        self.rep_array, self.word_array, self.o_array, _ = tf.while_loop(
            cond=termination_cond,
            body=tree_construction_body,
            loop_vars=(rep_array, word_array, o_array, 1),
            parallel_iterations=1
        )


