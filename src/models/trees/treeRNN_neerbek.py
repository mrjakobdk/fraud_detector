import tensorflow as tf
from utils.flags import FLAGS
from models.trees.treeModel import treeModel
import sys


class treeRNN_neerbek(treeModel):
    class_name = "treeRNN_neerbek"

    def build_constants(self):
        # embedding
        self.embeddings = tf.constant(self.data.word_embed_util.embeddings)
        ## dummi values
        self.rep_zero = tf.constant(0., shape=[FLAGS.sentence_embedding_size])
        self.word_zero = tf.constant(0., shape=[FLAGS.word_embedding_size])
        self.label_zero = tf.constant(0., shape=[FLAGS.label_size])

    def build_placeholders(self):
        # tree structure placeholders
        self.root_array = tf.placeholder(tf.int32, (None), name='root_array')
        self.is_leaf_array = tf.placeholder(tf.float32, (None, None), name='is_leaf_array')
        self.word_index_array = tf.placeholder(tf.int32, (None, None), name='word_index_array')
        self.left_child_array = tf.placeholder(tf.int32, (None, None), name='left_child_array')
        self.right_child_array = tf.placeholder(tf.int32, (None, None), name='right_child_array')
        self.label_array = tf.placeholder(tf.int32, (None, None, FLAGS.label_size), name='label_array')
        self.real_batch_size = tf.gather(tf.shape(self.is_leaf_array), 0)

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
                                   initializer=weight_initializer)
        self.U_R = tf.get_variable(name='U_R', shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                   initializer=weight_initializer)

        # bias
        self.b_W = tf.get_variable(name='b_W', shape=[FLAGS.sentence_embedding_size, 1], initializer=bias_initializer)
        self.b_U = tf.get_variable(name='b_U', shape=[FLAGS.sentence_embedding_size, 1], initializer=bias_initializer)

        # classifier weights
        self.V = tf.get_variable(name='V', shape=[FLAGS.label_size, FLAGS.sentence_embedding_size],
                                 initializer=xavier_initializer)
        self.b_p = tf.get_variable(name='b_p', shape=[FLAGS.label_size, 1], initializer=bias_initializer)

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
            right_child = tf.stack([tf.range(self.real_batch_size), tf.gather(self.right_child_array, i, axis=1)], axis=1)

            is_leaf_l = tf.reshape(tf.gather_nd(self.is_leaf_array, left_child), shape=(1, -1))
            is_leaf_r = tf.reshape(tf.gather_nd(self.is_leaf_array, right_child), shape=(1, -1))


            #todo this is not correct in the leaves!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            rep_l = gather_rep(i, self.left_child_array, rep_array)
            rep_r = gather_rep(i, self.right_child_array, rep_array)
            rep_word_l = gather_rep(i, self.left_child_array, word_array)
            rep_word_r = gather_rep(i, self.right_child_array, word_array)

            phrase_left = tf.matmul(self.U_L, rep_l) + tf.matmul(self.b_U, 1. - is_leaf_l)  # 1 - to negate is_leaf
            phrase_right = tf.matmul(self.U_R, rep_r) + tf.matmul(self.b_U, 1. - is_leaf_r)
            word_left = tf.matmul(self.W_L, rep_word_l) + tf.matmul(self.b_W, is_leaf_l)
            word_right = tf.matmul(self.W_R, rep_word_r) + tf.matmul(self.b_W, is_leaf_r)

            return tf.nn.leaky_relu(word_left + phrase_left + word_right + phrase_right)

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
            print_op = tf.print("o:", o,
                                output_stream=sys.stdout)
            with tf.control_dependencies([print_op]):
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

    def build_loss(self):
        logits = tf.gather_nd(tf.transpose(self.o_array.stack(), perm=[2, 0, 1]), self.root_array)  # roots_padded)
        labels = tf.gather_nd(self.label_array, self.root_array)

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

    def build_accuracy(self):
        # roots_pad = tf.constant([i for i in range(FLAGS.batch_size)])
        # roots_padded = tf.stack([roots_pad, self.root_array], axis=1)

        logits = tf.gather_nd(tf.transpose(self.o_array.stack(), perm=[2, 0, 1]), self.root_array)  # roots_padded)
        labels = tf.gather_nd(self.label_array, self.root_array)  # roots_padded)

        logits_max = tf.argmax(logits, axis=1)
        labels_max = tf.argmax(labels, axis=1)

        acc = tf.equal(logits_max, labels_max)
        self.acc = tf.reduce_mean(tf.cast(acc, tf.float32))
