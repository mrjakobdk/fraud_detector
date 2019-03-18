from models.trees.treeModel import treeModel
import tensorflow as tf
from utils.flags import FLAGS


class deepRNN(treeModel):

    def __init__(self, data, word_embed, model_placement, label_size=FLAGS.label_size, layers=FLAGS.deepRNN_layers):
        self.layers = layers
        super(deepRNN, self).__init__(data, word_embed, model_placement, label_size)

    def build_constants(self):
        # embedding
        self.embeddings = tf.constant(self.word_embed.embeddings)
        ## dummi values
        self.rep_zero = tf.constant(0., shape=[FLAGS.sentence_embedding_size])
        self.word_zero = tf.constant(0., shape=[FLAGS.word_embedding_size])
        self.label_zero = tf.constant(0., shape=[FLAGS.label_size])

    def build_placeholders(self):
        # tree structure placeholders
        self.root_array = tf.placeholder(tf.int32, (None),
                                         name='root_array')  # contains id for the root of each tree in the batch
        self.is_leaf_array = tf.placeholder(tf.float32, (None, None), name='is_leaf_array')
        self.word_index_array = tf.placeholder(tf.int32, (None, None), name='word_index_array')
        self.left_child_array = tf.placeholder(tf.int32, (None, None), name='left_child_array')
        self.right_child_array = tf.placeholder(tf.int32, (None, None), name='right_child_array')
        self.label_array = tf.placeholder(tf.int32, (None, None, self.label_size), name='label_array')
        self.real_batch_size = tf.gather(tf.shape(self.is_leaf_array), 0)

    def build_variables(self):
        xavier_initializer, weight_initializer, bias_initializer = self.get_initializers()

        # unique
        self.W_L = tf.get_variable(name='W_L', shape=[FLAGS.sentence_embedding_size, FLAGS.word_embedding_size],
                                   initializer=xavier_initializer)
        self.W_R = tf.get_variable(name='W_R', shape=[FLAGS.sentence_embedding_size, FLAGS.word_embedding_size],
                                   initializer=xavier_initializer)

        self.E_2Leaf = tf.get_variable(name='E_2Leaf', shape=[FLAGS.sentence_embedding_size, FLAGS.word_embedding_size],
                                       initializer=xavier_initializer)

        # repeated for each tree layer
        self.b = []
        self.E = []
        self.U_L = []
        self.U_R = []
        for i in range(self.layers):
            self.b.append(
                tf.get_variable(name='b-' + str(i), shape=[FLAGS.sentence_embedding_size, 1],
                                initializer=bias_initializer))
            self.E.append(
                tf.get_variable(name='E-' + str(i),
                                shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                initializer=weight_initializer))
            self.U_L.append(
                tf.get_variable(name='U_L-' + str(i),
                                shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                initializer=weight_initializer))
            self.U_R.append(
                tf.get_variable(name='U_R-' + str(i),
                                shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                initializer=weight_initializer))

        # classifier weights
        self.V = tf.get_variable(name='V', shape=[FLAGS.label_size, FLAGS.sentence_embedding_size],
                                 initializer=xavier_initializer)
        self.b_p = tf.get_variable(name='b_p', shape=[FLAGS.label_size, 1], initializer=bias_initializer)

    def build_model(self):
        rep_arrays = []
        for i in range(self.layers):
            rep_array = tf.TensorArray(
                tf.float32,
                size=0,
                dynamic_size=True,
                clear_after_read=False,
                infer_shape=False)
            rep_array = rep_array.write(0, tf.reshape(tf.tile(self.rep_zero, tf.stack([self.real_batch_size])),
                                                      [FLAGS.sentence_embedding_size, -1]))
            rep_arrays.append(rep_array)

        word_array = tf.TensorArray(
            tf.float32,
            size=0,
            dynamic_size=True,
            clear_after_read=False,
            infer_shape=False)
        word_array = word_array.write(0, tf.reshape(tf.tile(self.word_zero, tf.stack([self.real_batch_size])),
                                                    [FLAGS.word_embedding_size, -1]))

        batch_indices = tf.stack([tf.range(self.real_batch_size), tf.range(self.real_batch_size)], axis=1)

        def gather_rep(step, children_indices, rep_array):
            children = tf.squeeze(tf.gather(children_indices, step, axis=1))
            rep_entries = rep_array.gather(children)
            t_rep_entries = tf.transpose(rep_entries, perm=[0, 2, 1])
            return tf.transpose(tf.gather_nd(t_rep_entries, batch_indices))

        # tree layer 0
        def embed_word(word_index):
            return tf.transpose(tf.nn.embedding_lookup(self.embeddings, word_index))

        def build_node_layer_0(i, rep_array, word_array):
            is_leaf = tf.reshape(tf.gather(self.is_leaf_array, i, axis=1), shape=(1, -1))

            # print_op = tf.print("is_leaf:", is_leaf, "b^0", self.b[0],
            #                     output_stream=sys.stdout)
            # with tf.control_dependencies([print_op]):
            rep_l = gather_rep(i, self.left_child_array, rep_array)
            rep_r = gather_rep(i, self.right_child_array, rep_array)
            rep_word_l = gather_rep(i, self.left_child_array, word_array)
            rep_word_r = gather_rep(i, self.right_child_array, word_array)

            phrase_left = tf.matmul(self.U_L[0], rep_l)
            phrase_right = tf.matmul(self.U_R[0], rep_r)
            word_left = tf.matmul(self.W_L, rep_word_l)
            word_right = tf.matmul(self.W_R, rep_word_r)

            return tf.nn.leaky_relu(word_left
                                    + word_right
                                    + phrase_left
                                    + phrase_right
                                    + tf.matmul(self.b[0], is_leaf))

        def tree_layer_0_construction_body(rep_array, word_array, i):
            word_index = tf.gather(self.word_index_array, i, axis=1)
            word_emb = embed_word(word_index)
            word_array = word_array.write(i, word_emb)

            rep = build_node_layer_0(i, rep_array, word_array)
            rep_array = rep_array.write(i, rep)

            i = tf.add(i, 1)
            return rep_array, word_array, i

        termination_cond = lambda a, b, i: tf.less(i, tf.gather(tf.shape(self.word_index_array), 1))
        rep_arrays[0], word_array, _ = tf.while_loop(
            cond=termination_cond,
            body=tree_layer_0_construction_body,
            loop_vars=(rep_arrays[0], word_array, 1),
            parallel_iterations=1
        )

        # tree layer 1
        if self.layers > 1:
            def build_node_layer_1(i, rep_array, rep_array_prev, word_array):
                rep_l = gather_rep(i, self.left_child_array, rep_array)
                rep_r = gather_rep(i, self.right_child_array, rep_array)
                rep_prev = rep_array_prev.read(i)
                rep_word = word_array.read(i)

                phrase_left = tf.matmul(self.U_L[1], rep_l)
                phrase_right = tf.matmul(self.U_R[1], rep_r)
                word_encode = tf.matmul(self.E_2Leaf, rep_word)
                phrase_encode = tf.matmul(self.E[1], rep_prev)

                return tf.nn.leaky_relu(phrase_left + phrase_right + word_encode + phrase_encode + self.b[1])

            def tree_layer_1_construction_body(rep_array, rep_array_prev, word_array, i):
                rep = build_node_layer_1(i, rep_array, rep_array_prev, word_array)
                rep_array = rep_array.write(i, rep)

                i = tf.add(i, 1)
                return rep_array, rep_array_prev, word_array, i

            termination_cond = lambda a, b, c, i: tf.less(i, tf.gather(tf.shape(self.word_index_array), 1))
            rep_arrays[1], rep_arrays[0], word_array, _ = tf.while_loop(
                cond=termination_cond,
                body=tree_layer_1_construction_body,
                loop_vars=(rep_arrays[1], rep_arrays[0], word_array, 1),
                parallel_iterations=1
            )

        # tree layer 2-n
        for layer in range(2, self.layers):
            def build_node_layer_n(i, rep_array, rep_array_prev, U_L, U_R, E, b):
                rep_l = gather_rep(i, self.left_child_array, rep_array)
                rep_r = gather_rep(i, self.right_child_array, rep_array)
                rep_prev = rep_array_prev.read(i)

                phrase_left = tf.matmul(U_L, rep_l)
                phrase_right = tf.matmul(U_R, rep_r)
                phrase_encode = tf.matmul(E, rep_prev)

                return tf.nn.leaky_relu(phrase_left + phrase_right + phrase_encode + b)

            def tree_layer_n_construction_body(rep_array, rep_array_prev, U_L, U_R, E, b, i):
                rep = build_node_layer_n(i, rep_array, rep_array_prev, U_L, U_R, E, b)
                rep_array = rep_array.write(i, rep)

                i = tf.add(i, 1)
                return rep_array, rep_array_prev, U_L, U_R, E, b, i

            termination_cond = lambda a, b, c, d, e, f, i: tf.less(i, tf.gather(tf.shape(self.word_index_array), 1))
            rep_arrays[layer], rep_arrays[layer - 1], _, _, _, _, _ = tf.while_loop(
                cond=termination_cond,
                body=tree_layer_n_construction_body,
                loop_vars=(rep_arrays[layer], rep_arrays[layer - 1], self.U_L[layer], self.U_R[layer], self.E[layer],
                           self.b[layer], 1),
                parallel_iterations=1
            )

        # softmax layer
        last_tree_layer = rep_arrays[self.layers - 1]
        root_rep = tf.gather_nd(tf.transpose(last_tree_layer.stack(), perm=[2, 0, 1]), self.root_array)
        self.logits = tf.transpose(tf.matmul(self.V, tf.transpose(root_rep)) + self.b_p)

    def build_loss(self):
        # print_op = tf.print("root:", self.root_array,
        #                     output_stream=sys.stdout)
        # with tf.control_dependencies([print_op]):
        labels = tf.gather_nd(self.label_array, self.root_array)
        softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=labels)
        self.loss = tf.reduce_mean(softmax_cross_entropy)

    def build_accuracy(self):
        labels = tf.gather_nd(self.label_array, self.root_array)

        logits_max = tf.argmax(self.logits, axis=1)
        labels_max = tf.argmax(labels, axis=1)

        acc = tf.equal(logits_max, labels_max)
        self.acc = tf.reduce_mean(tf.cast(acc, tf.float32))

    def build_predict(self):
        self.p = tf.nn.softmax(self.logits, axis=-1)
        self.labels = tf.gather_nd(self.label_array, self.root_array)