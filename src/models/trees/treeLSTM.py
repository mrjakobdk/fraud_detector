import tensorflow as tf
from utils.flags import FLAGS
from models.trees.treeModel import treeModel


class treeLSTM(treeModel):

    def build_constants(self):
        # embedding
        self.embeddings = tf.constant(self.word_embed.embeddings)
        ## dummi values
        self.rep_zero = tf.constant(0., shape=[FLAGS.sentence_embedding_size])
        self.word_zero = tf.constant(0., shape=[FLAGS.word_embedding_size, 1])
        self.label_zero = tf.constant(0., shape=[FLAGS.label_size])

    # def build_placeholders(self):
    #     # tree structure placeholders
    #     self.root_array = tf.placeholder(tf.int32, (None), name='root_array')
    #     self.is_leaf_array = tf.placeholder(tf.bool, (None, None), name='is_leaf_array')
    #     self.word_index_array = tf.placeholder(tf.int32, (None, None), name='word_index_array')
    #     self.left_child_array = tf.placeholder(tf.int32, (None, None), name='left_child_array')
    #     self.right_child_array = tf.placeholder(tf.int32, (None, None), name='right_child_array')
    #     self.label_array = tf.placeholder(tf.int32, (None, None, FLAGS.label_size), name='label_array')
    #     self.real_batch_size = tf.gather(tf.shape(self.is_leaf_array), 0)

    def build_variables(self):
        # initializers
        xavier_initializer, weight_initializer, bias_initializer = self.get_initializers()

        # word variables
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
                                    initializer=weight_initializer)
        self.Ui_R = tf.get_variable(name='Ui_R', shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                    initializer=weight_initializer)
        self.Uf_L = tf.get_variable(name='Uf_L', shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                    initializer=weight_initializer)
        self.Uf_R = tf.get_variable(name='Uf_R', shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                    initializer=weight_initializer)
        self.Uo_L = tf.get_variable(name='Uo_L', shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                    initializer=weight_initializer)
        self.Uo_R = tf.get_variable(name='Uo_R', shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                    initializer=weight_initializer)
        self.Uc_L = tf.get_variable(name='Uc_L', shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                    initializer=weight_initializer)
        self.Uc_R = tf.get_variable(name='Uc_R', shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                                    initializer=weight_initializer)

        # bias
        self.bi = tf.get_variable(name='bi', shape=[FLAGS.sentence_embedding_size, 1], initializer=bias_initializer)
        self.bf = tf.get_variable(name='bf', shape=[FLAGS.sentence_embedding_size, 1], initializer=bias_initializer)
        self.bo = tf.get_variable(name='bo', shape=[FLAGS.sentence_embedding_size, 1], initializer=bias_initializer)
        self.bc = tf.get_variable(name='bc', shape=[FLAGS.sentence_embedding_size, 1], initializer=bias_initializer)

        # classifier weights
        self.V = tf.get_variable(name='V', shape=[FLAGS.label_size, FLAGS.sentence_embedding_size],
                                 initializer=xavier_initializer)
        self.b_p = tf.get_variable(name='b_p', shape=[FLAGS.label_size, 1], initializer=bias_initializer)

    def build_model(self):
        mem_array = tf.TensorArray(
            tf.float32,
            size=0,
            dynamic_size=True,
            clear_after_read=False,
            infer_shape=False)
        mem_array = mem_array.write(0, tf.reshape(tf.tile(self.rep_zero, tf.stack([self.real_batch_size])),
                                                  [FLAGS.sentence_embedding_size, -1]))

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

        def build_node(i, rep_array, word_array, mem_array):
            rep_l = gather_rep(i, self.left_child_array, rep_array)
            rep_r = gather_rep(i, self.right_child_array, rep_array)
            c_l = gather_rep(i, self.left_child_array, mem_array)
            c_r = gather_rep(i, self.right_child_array, mem_array)
            rep_word = word_array.read(i)

            i_n = tf.sigmoid(
                tf.matmul(self.Wi, rep_word) + tf.matmul(self.Ui_L, rep_l) + tf.matmul(self.Ui_R, rep_r) + self.bi)

            f_l = tf.sigmoid(
                tf.matmul(self.Wf, rep_word) + tf.matmul(self.Uf_R, rep_l) + self.bf)

            f_r = tf.sigmoid(
                tf.matmul(self.Wf, rep_word) + tf.matmul(self.Uf_L, rep_r) + self.bf)

            o_n = tf.sigmoid(
                tf.matmul(self.Wo, rep_word) + tf.matmul(self.Uo_L, rep_l) + tf.matmul(self.Uo_R, rep_r) + self.bo)

            u_n = tf.tanh(
                tf.matmul(self.Wc, rep_word) + tf.matmul(self.Uc_L, rep_l) + tf.matmul(self.Uc_R, rep_r) + self.bc)

            c_n = tf.math.multiply(i_n, u_n) + tf.math.multiply(f_l, c_l) + tf.math.multiply(f_r, c_r)
            h_n = tf.math.multiply(o_n, tf.tanh(c_n))

            return h_n, c_n

        def tree_construction_body(rep_array, word_array, mem_array, o_array, i):
            word_index = tf.gather(self.word_index_array, i, axis=1)
            # print_op = tf.print("word_index:", word_index,
            #                     output_stream=sys.stdout)
            # with tf.control_dependencies([print_op]):
            word_emb = embed_word(word_index)
            word_array = word_array.write(i, word_emb)

            rep, c = build_node(i, rep_array, word_array, mem_array)
            rep_array = rep_array.write(i, rep)
            mem_array = mem_array.write(i, c)

            o = tf.matmul(self.V, rep) + self.b_p
            o_array = o_array.write(i, o)

            i = tf.add(i, 1)
            return rep_array, word_array, mem_array, o_array, i

        termination_cond = lambda rep_a, word_a, m_a, o_a, i: tf.less(i, tf.gather(tf.shape(self.left_child_array), 1))

        self.rep_array, self.word_array, self.mem_array, self.o_array, _ = tf.while_loop(
            cond=termination_cond,
            body=tree_construction_body,
            loop_vars=(rep_array, word_array, mem_array, o_array, 1),
            parallel_iterations=1
        )

    # def build_loss(self):
    #     logits = tf.gather_nd(tf.transpose(self.o_array.stack(), perm=[2, 0, 1]), self.root_array)  # roots_padded)
    #     labels = tf.gather_nd(self.label_array, self.root_array)
    #
    #     softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
    #     self.loss = tf.reduce_mean(softmax_cross_entropy)
    #
    #     # reg_weight = 0.001
    #     # self.loss += reg_weight * tf.nn.l2_loss(self.W)
    #     # self.loss += reg_weight * tf.nn.l2_loss(self.U_L)
    #     # self.loss += reg_weight * tf.nn.l2_loss(self.U_R)
    #     # self.loss += reg_weight * tf.nn.l2_loss(self.V)
    #
    #     # todo is this reshaped in the correct way?
    #     # todo fix loss - might drag the bias to zero??? tests and fix it
    #     # self.loss = tf.reduce_mean(
    #     #     tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.reshape(self.o_array.stack(), [-1, FLAGS.label_size]),
    #     #                                                # stacking o_array this way might be wrong
    #     #                                                labels=self.label_array))
    #
    # def build_accuracy(self):
    #     # roots_pad = tf.constant([i for i in range(FLAGS.batch_size)])
    #     # roots_padded = tf.stack([roots_pad, self.root_array], axis=1)
    #
    #     logits = tf.gather_nd(tf.transpose(self.o_array.stack(), perm=[2, 0, 1]), self.root_array)  # roots_padded)
    #     labels = tf.gather_nd(self.label_array, self.root_array)  # roots_padded)
    #
    #     logits_max = tf.argmax(logits, axis=1)
    #     labels_max = tf.argmax(labels, axis=1)
    #
    #     acc = tf.equal(logits_max, labels_max)
    #     self.acc = tf.reduce_mean(tf.cast(acc, tf.float32))
