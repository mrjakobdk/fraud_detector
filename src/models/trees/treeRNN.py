import tensorflow as tf
from utils.flags import FLAGS
import utils.helper as helper
import utils.tree_util as tree_util
import sys
from models.trees.treeModel import treeModel


class treeRNN(treeModel):

    def build_constants(self):
        # Setup data
        self.embeddings = tf.constant(self.word_embed.embeddings)

        # constants
        # leaf constant output
        self.o_none = tf.constant(-1.0, shape=[FLAGS.label_size, 1])

    def build_placeholders(self):
        # tree structure placeholders
        self.root_array = tf.placeholder(tf.int32, (None), name='root_array')
        self.is_leaf_array = tf.placeholder(tf.bool, (None), name='is_leaf_array')
        self.word_index_array = tf.placeholder(tf.int32, (None), name='word_index_array')
        self.left_child_array = tf.placeholder(tf.int32, (None), name='left_child_array')
        self.right_child_array = tf.placeholder(tf.int32, (None), name='right_child_array')
        self.label_array = tf.placeholder(tf.int32, (None, FLAGS.label_size), name='label_array')

    def build_variables(self):
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
        self.W_l = tf.get_variable(name='W_l', shape=[FLAGS.sentence_embedding_size, FLAGS.word_embedding_size],
                                   initializer=weight_initializer)
        self.W_r = tf.get_variable(name='W_r', shape=[FLAGS.sentence_embedding_size, FLAGS.word_embedding_size],
                                   initializer=weight_initializer)

        # phrase weights
        self.U_l = tf.get_variable(name='U_l', shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                   initializer=weight_initializer)
        self.U_r = tf.get_variable(name='U_r', shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                   initializer=weight_initializer)

        # bias
        self.b_W = tf.get_variable(name='b_W', shape=[FLAGS.sentence_embedding_size, 1], initializer=bias_initializer)
        self.b_U = tf.get_variable(name='b_U', shape=[FLAGS.sentence_embedding_size, 1], initializer=bias_initializer)

        # classifier weights
        self.V = tf.get_variable(name='V', shape=[FLAGS.label_size, FLAGS.sentence_embedding_size],
                                 initializer=xavier_initializer)
        self.b_p = tf.get_variable(name='b_p', shape=[FLAGS.label_size, 1], initializer=bias_initializer)

    def build_model(self):

        helper._print_header("Constructing tRNN structure")

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

        helper._print_header("Building tRNN tree structure")

        # build the tRNN structure
        def embed_word(word_index):
            return tf.nn.embedding_lookup(self.embeddings, word_index)

        def build_node(left_child, right_child, rep_array):
            left_is_leaf = tf.gather(self.is_leaf_array, left_child)
            right_is_leaf = tf.gather(self.is_leaf_array, right_child)

            # reshape from vector to matrix with height 300 and width 1
            rep_l = tf.reshape(rep_array.read(left_child), [300, 1])
            rep_r = tf.reshape(rep_array.read(right_child), [300, 1])

            left = tf.cond(
                left_is_leaf,
                lambda: tf.matmul(self.W_l, rep_l) + self.b_W,
                lambda: tf.matmul(self.U_l, rep_l) + self.b_U
            )

            right = tf.cond(
                right_is_leaf,
                lambda: tf.matmul(self.W_r, rep_r) + self.b_W,
                lambda: tf.matmul(self.U_r, rep_r) + self.b_U
            )

            # relu( (sent_size , 1) + (sent_size , 1) + (sent_size , 1) )  = (sent_size , 1)
            return tf.nn.leaky_relu(left + right)

        def tree_construction_body(rep_array, o_array, i):
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
                lambda: build_node(left_child, right_child, rep_array)
            )
            rep_array = rep_array.write(i, rep)

            # o_none = (label_size, 1)
            # softmax( (label_size, sent_size) * (sent_size, 1) + (label_size, 1)) = (label_size, 1)
            o = tf.cond(
                is_leaf,
                lambda: self.o_none,
                lambda: tf.matmul(self.V, rep) + self.b_p  # TODO maybe with out activation function
            )
            o_array = o_array.write(i, o)

            i = tf.add(i, 1)
            return rep_array, o_array, i

        termination_cond = lambda rep_a, o_a, i: tf.less(i, tf.squeeze(tf.shape(self.is_leaf_array)))

        tf.print('hello', (self.is_leaf_array), output_stream=sys.stderr)

        self.rep_array, self.o_array, _ = tf.while_loop(
            cond=termination_cond,
            body=tree_construction_body,
            loop_vars=(rep_array, o_array, 0),
            parallel_iterations=1
        )

    def build_loss(self):
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.reshape(self.o_array.stack(), [-1, FLAGS.label_size]),
                                                       labels=self.label_array))

    def build_accuracy(self):
        # Accuracy
        root_array = self.root_array

        logists = tf.reshape(self.o_array.gather(root_array), [-1, FLAGS.label_size])
        labels = tf.gather(self.label_array, root_array)

        o_max = tf.reshape(
            tf.argmax(
                logists, axis=1), [-1])

        label_max = tf.argmax(
            labels, axis=1)

        acc = tf.equal(
            o_max,
            label_max)
        self.acc = tf.reduce_mean(tf.cast(acc, tf.float32))


    def build_feed_dict(self, batch):
        node_list_list = []
        node_to_index_list = []
        root_array = []
        last_root = -1
        for root in batch:
            node_list = []
            tree_util.depth_first_traverse(root, node_list, lambda node, node_list: node_list.append(node))
            node_list_list.append(node_list)
            node_to_index = helper.reverse_dict(node_list)
            node_to_index_list.append(node_to_index)
            last_root += tree_util.size_of_tree(root)
            root_array.append(last_root)

        feed_dict = {
            self.root_array: root_array,
            self.is_leaf_array: helper.flatten([[node.is_leaf for node in node_list] for node_list in node_list_list]),
            self.word_index_array: helper.flatten([[self.word_embed.get_idx(node.value) for node in node_list]
                                                   for node_list in node_list_list]),
            self.left_child_array: helper.flatten([
                [node_to_index[node.left_child] if node.left_child is not None else -1 for node in node_list]
                for node_list, node_to_index in zip(node_list_list, node_to_index_list)]),
            self.right_child_array: helper.flatten([
                [node_to_index[node.right_child] if node.right_child is not None else -1 for node in node_list]
                for node_list, node_to_index in zip(node_list_list, node_to_index_list)]),
            self.label_array: helper.flatten([[node.label for node in node_list]
                                              for node_list in node_list_list])
        }

        return feed_dict
