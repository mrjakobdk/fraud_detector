import tensorflow as tf
from utils.flags import FLAGS
import utils.helper as helper


class tRNN:
    def __init__(self, embeddings):
        helper._print("========= Constructing tRNN placeholders and variables =========")

        # TODO: Add flag for loss weight

        # tree structure placeholders
        self.is_leaf_array = tf.placeholder(tf.bool, (None), name='is_leaf_array')
        self.word_index_array = tf.placeholder(tf.int32, (None), name='word_index_array')
        self.left_child_array = tf.placeholder(tf.int32, (None), name='left_child_array')
        self.right_child_array = tf.placeholder(tf.int32, (None), name='right_child_array')

        # encoding variables
        # word weights
        W_l = tf.get_variable(name='W_l', shape=[FLAGS.word_embedding_size, FLAGS.sentence_embedding_size])
        W_r = tf.get_variable(name='W_r', shape=[FLAGS.word_embedding_size, FLAGS.sentence_embedding_size])

        # phrase weights
        U_l = tf.get_variable(name='U_l', shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size])
        U_r = tf.get_variable(name='U_r', shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size])

        # bias
        b = tf.get_variable(name='b', shape=[1, FLAGS.sentence_embedding_size])

        # classifier weights
        V = tf.get_variable(name='V', shape=[FLAGS.sentence_embedding_size, FLAGS.labels])
        b_p = tf.get_variable(name='b_p', shape=[1, FLAGS.label_size])

        #leaf constant output
        o_none = tf.constant(-1.0, shape=[1, FLAGS.label_size])

        # phrase node tensors
        rep_array = tf.TensorArray(
            tf.float32,
            size=0,
            dynamic_size=True,
            clear_after_read=False,
            infer_shape=False)

        o_array = tf.TensorArray(
            tf.float32,
            size=0,
            dynamic_size=True,
            clear_after_read=False,
            infer_shape=False)

        helper._print("========= Building tRNN tree structure =========")

        # build the tRNN structure
        def embed_word(word_index):
            return tf.nn.embedding_lookup(embeddings, word_index)

        def build_node(left_child, right_child, rep_array):
            left_is_leaf = tf.gather(self.is_leaf_array, left_child)
            right_is_leaf = tf.gather(self.is_leaf_array, right_child)

            rep_l = rep_array.read(left_child)
            rep_r = rep_array.read(right_child)

            left = tf.cond(
                left_is_leaf,
                lambda: tf.matmul(W_l, rep_l),
                lambda: tf.matmul(U_l, rep_l)
            )

            right = tf.cond(
                right_is_leaf,
                lambda: tf.matmul(W_r, rep_r),
                lambda: tf.matmul(U_r, rep_r)
            )

            return tf.nn.relu(left + right + b)

        def tree_construction_body(rep_array, o_array, i):
            # gather variables
            is_leaf = tf.gather(self.is_leaf_array, i)
            word_index = tf.gather(self.word_index_array, i)
            left_child = tf.gather(self.left_child_array, i)
            right_child = tf.gather(self.right_child_array, i)

            rep = tf.cond(
                is_leaf,
                lambda: embed_word(word_index),
                lambda: build_node(left_child, right_child, rep_array)
            )
            rep_array.write(i, rep)

            o = tf.cond(
                is_leaf,
                lambda: o_none,
                lambda: tf.nn.softmax(tf.matmul(V, rep) + b_p)#TODO maybe with out activation function
            )
            o_array.write(i, o)

            i = tf.add(i, 1)
            return rep_array, o_array,  i

        termination_cond = lambda rep_array, o_array, i: tf.less(i, tf.squeeze(tf.shape(self.is_leaf_array)))

        self.rep_array, self.o_array, _ = tf.while_loop(
            cond=termination_cond,
            body=tree_construction_body,
            loop_vars=[rep_array, o_array, 0],
            parallel_iterations=1
        )

    def get_rep_array(self):
        return self.rep_array

    def get_root_tensor(self):
        return self.rep_array.read(tf.squeeze(tf.shape(self.is_leaf_array)))

    def loss(self): #TODO you ended here

        tf.nn.matmul(self.weight, tf.nn.matmul(self.label_array, tf.log(self.o_array)))

        loss = tf.reduce_sum()

        return loss