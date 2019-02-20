import tensorflow as tf
from utils.flags import FLAGS
import utils.helper as helper
import utils.tree_util as tree_util
import numpy as np
import sys
import os
import webbrowser
import time


class tRNN:
    feed_dict_test_list = []

    def __init__(self, data):
        """
        :param data: utils.data
        """
        helper._print("========= Constructing tRNN constants, placeholders and variables =========")

        # Setup data
        self.data = data  # TODO: Make data
        self.embeddings = tf.constant(data.word_embed_util.embeddings)

        # constants
        # leaf constant output
        o_none = tf.constant(-1.0, shape=[FLAGS.label_size, 1])
        # loss weight constant w>1 more weight on sensitive loss
        self.weight = tf.constant(FLAGS.sensitive_weight)

        # tree structure placeholders
        self.is_leaf_array = tf.placeholder(tf.bool, (None), name='is_leaf_array')
        self.word_index_array = tf.placeholder(tf.int32, (None), name='word_index_array')
        self.left_child_array = tf.placeholder(tf.int32, (None), name='left_child_array')
        self.right_child_array = tf.placeholder(tf.int32, (None), name='right_child_array')
        self.label_array = tf.placeholder(tf.int32, (None, FLAGS.label_size), name='label_array')

        # initializers
        W_max = np.sqrt(6. / (FLAGS.word_embedding_size + FLAGS.sentence_embedding_size))
        U_max = np.sqrt(6. / (FLAGS.sentence_embedding_size + FLAGS.sentence_embedding_size))
        V_max = np.sqrt(6. / (FLAGS.sentence_embedding_size + FLAGS.label_size))
        b_max = np.sqrt(6. / FLAGS.sentence_embedding_size)
        b_p_max = 0  # todo np.sqrt(6. / FLAGS.label_size)
        W_initializer = tf.initializers.random_uniform(minval=-W_max, maxval=W_max)
        U_initializer = tf.initializers.random_uniform(minval=-U_max, maxval=U_max)  # todo why this?
        V_initializer = tf.initializers.random_uniform(minval=-V_max, maxval=V_max)
        b_initializer = tf.initializers.random_uniform(minval=-b_max, maxval=b_max)
        b_p_initializer = tf.initializers.random_uniform(minval=-b_p_max, maxval=b_p_max)

        # encoding variables
        W_l = tf.get_variable(name='W_l', shape=[FLAGS.sentence_embedding_size, FLAGS.word_embedding_size],
                              initializer=W_initializer)
        W_r = tf.get_variable(name='W_r', shape=[FLAGS.sentence_embedding_size, FLAGS.word_embedding_size],
                              initializer=W_initializer)
        self.W_l = W_l
        self.W_r = W_r

        # phrase weights
        U_l = tf.get_variable(name='U_l', shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                              initializer=U_initializer)
        U_r = tf.get_variable(name='U_r', shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                              initializer=U_initializer)
        self.U_l = U_l
        self.U_r = U_r

        # bias
        b_W = tf.get_variable(name='b_W', shape=[FLAGS.sentence_embedding_size, 1], initializer=b_initializer)
        b_U = tf.get_variable(name='b_U', shape=[FLAGS.sentence_embedding_size, 1], initializer=b_initializer)

        # classifier weights
        V = tf.get_variable(name='V', shape=[FLAGS.label_size, FLAGS.sentence_embedding_size],
                            initializer=V_initializer)
        b_p = tf.get_variable(name='b_p', shape=[FLAGS.label_size, 1], initializer=b_p_initializer)
        self.V = V
        self.b_p = b_p

        helper._print("========= Constructing tRNN structure =========")

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
            return tf.nn.embedding_lookup(self.embeddings, word_index)

        def build_node(left_child, right_child, rep_array):
            left_is_leaf = tf.gather(self.is_leaf_array, left_child)
            right_is_leaf = tf.gather(self.is_leaf_array, right_child)

            # reshape from vector to matrix with height 300 and width 1
            rep_l = tf.reshape(rep_array.read(left_child), [300, 1])
            rep_r = tf.reshape(rep_array.read(right_child), [300, 1])

            # print_op_l = tf.print("build_node left W:",W_l.shape, ' U:', U_l.shape, ' rep:', rep_l.shape, '\n',
            #                     output_stream=sys.stdout)
            #
            # print_op_r = tf.print("build_node left W:",W_r.shape, ' U:', U_r.shape, ' rep:', rep_r.shape, '\n',
            #                     output_stream=sys.stdout)
            #
            # # (sent_size, sent_size) * (sent_size, 1) = (sent_size, 1)
            # with tf.control_dependencies([print_op_l, print_op_r]):
            left = tf.cond(
                left_is_leaf,
                lambda: tf.matmul(W_l, rep_l) + b_W,
                lambda: tf.matmul(U_l, rep_l) + b_U
            )

            right = tf.cond(
                right_is_leaf,
                lambda: tf.matmul(W_r, rep_r) + b_W,
                lambda: tf.matmul(U_r, rep_r) + b_U
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
                lambda: o_none,
                lambda: tf.matmul(V, rep) + b_p  # TODO maybe with out activation function
            )
            # print_op = tf.print("o:", o, "rep:", rep,
            #                     # ' original array and stack():', self.o_array, self.o_array.stack(),
            #                     output_stream=sys.stdout)
            # with tf.control_dependencies([print_op]):
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

        self.loss = self.get_loss()
        self.acc = self.get_acc()

        self.train_op = tf.train.AdagradOptimizer(FLAGS.learning_rate).minimize(self.loss)
        self.init = tf.global_variables_initializer()

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.acc)
        self.merged_summary_op = tf.summary.merge_all()

    def get_acc_old(self):
        # Accuracy
        o_max = tf.reshape(tf.argmax(
            self.o_array.stack(), axis=1), [-1])

        label_max = tf.argmax(
            self.label_array, axis=1)

        acc = tf.equal(
            o_max,
            label_max)
        acc = tf.reduce_mean(tf.cast(acc, tf.float32))
        tf.summary.scalar("accuracy", acc)
        return acc

    def get_acc(self):

        # Accuracy
        root_index = self.o_array.size() - 1

        o_max = tf.reshape(
            tf.argmax(
                self.o_array.read(root_index)), [-1])

        label_max = tf.argmax(
            tf.gather(self.label_array, root_index))

        acc = tf.equal(
            o_max,
            label_max)
        acc = tf.reduce_mean(tf.cast(acc, tf.float32))
        tf.summary.scalar("accuracy", acc)
        return acc

    def get_loss(self):
        root_index = self.o_array.size() - 1
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=tf.reshape(self.o_array.read(root_index), [-1, FLAGS.label_size]),
                labels=tf.gather(self.label_array, root_index)))
        return loss

    def get_loss_old(self):
        # todo change to the correct loss
        # Loss
        # pro_1 = tf.matmul(self.weight,
        #                   tf.matmul(self.label_array,
        #                             tf.log(self.o_array.concat())))
        # pro_2 = tf.matmul(1 - self.label_array, tf.log(1 - self.o_array.concat()))
        # loss = - tf.reduce_sum(pro_1 + pro_2)
        # tf.summary.scalar("loss", loss)
        # Todo: check reshape.

        weights = tf.constant([0.6646, 1 / 0.6646])
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.reshape(self.o_array.stack(), [-1, FLAGS.label_size]),
                                                       labels=self.label_array))
        # tf.nn.weighted_cross_entropy_with_logits(logits=tf.reshape(self.o_array.stack(), [-1, FLAGS.label_size]),
        #                                          targets=self.label_array,
        #                                          pos_weight=weights))#todo not the corret one should be softmax, this is sigmoid
        # loss += tf.nn.l2_loss(self.W_l) + tf.nn.l2_loss(self.W_r) + tf.nn.l2_loss(self.U_l) + tf.nn.l2_loss(
        #     self.U_r) + tf.nn.l2_loss(self.V) #TODO what went wrong

        return loss

    def train(self):
        helper._print("========= Training tRNN =========")
        helper._print("Test ration:", tree_util.ratio_of_labels(self.data.test_trees))

        # todo make a flag for this
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )

        with tf.Session(config=config) as sess:

            # Summary writer for both the training and the set acc and loss - used for tensorboard
            train_writer = tf.summary.FileWriter(FLAGS.logs_dir + 'rnn/train', sess.graph)
            test_writer = tf.summary.FileWriter(FLAGS.logs_dir + 'rnn/test')

            loss_total = 0
            acc_total = 0

            # Run the init
            sess.run(self.init)
            saver = tf.train.Saver()

            self.run_tensorboard()
            start_time = time.time()
            for epoch in range(FLAGS.epochs):
                helper._print("========= Epoch ", epoch + 1, ' =========')
                best_acc = 0
                helper._print("b_p:", sess.run(self.b_p))
                for step, tree in enumerate(np.random.permutation(self.data.train_trees)):  # todo build train get_trees

                    if step % FLAGS.print_step_interval == 0:
                        test_acc, test_loss, test_time = self.compute_acc_loss(self.data.test_trees, sess, test_writer,
                                                                               epoch * len(
                                                                                   self.data.train_trees) + step)

                        avg_acc = acc_total / FLAGS.print_step_interval
                        avg_loss = loss_total / FLAGS.print_step_interval
                        summary = tf.Summary()
                        summary.value.add(tag='accuracy', simple_value=avg_acc)
                        summary.value.add(tag='loss', simple_value=avg_loss)
                        train_writer.add_summary(summary, epoch * len(self.data.train_trees) + step)
                        loss_total = 0
                        acc_total = 0

                        helper._print("Epoch:", epoch + 1, "Step:",
                                      step)  # todo avg does not say much maybe eval on validation
                        helper._print("Test -  acc:", test_acc, "loss:", test_loss, "time:", test_time)
                        helper._print("Train -  acc:", avg_acc, "loss:", avg_loss)

                        if test_acc > best_acc:  # TODO should be replaced with validation set
                            saver.save(sess, FLAGS.model_filename)  # TODO create flag

                    feed_dict = self.build_feed_dict(tree)  # todo maybe change to batches
                    _, acc, loss = sess.run([self.train_op, self.acc, self.loss],
                                          feed_dict=feed_dict)

                    acc_total += acc
                    loss_total += loss

                    #train_writer.add_summary(summary, epoch * len(self.data.train_trees) + step)



                helper._print("Avg Epoch Time:", (time.time() - start_time) / (epoch + 1) / 60, "m")


    def compute_acc_loss_old(self, data, sess, summary_writer, steps):
        start = time.time()
        loss_total = 0
        acc_total = 0
        for step, tree in enumerate(data):
            feed_dict = self.build_feed_dict(tree)
            acc, loss, summary = sess.run([self.acc, self.loss, self.merged_summary_op],
                                          feed_dict=feed_dict)  # todo fix summary
            acc_total += acc
            loss_total += loss
            summary_writer.add_summary(summary, steps)
        end = time.time()

        return acc_total / len(data), loss_total / len(data), end - start

    def compute_acc_loss(self, data, sess, summary_writer, steps):#todo should only be for test
        start = time.time()
        loss_total = 0
        acc_total = 0

        if len(self.feed_dict_test_list)==0:
            for tree in data:
                self.feed_dict_test_list.append(self.build_feed_dict(tree))
        for step, feed_dict in enumerate(self.feed_dict_test_list):
            acc, loss = sess.run([self.acc, self.loss],
                                          feed_dict=feed_dict)  # todo fix summary
            acc_total += acc
            loss_total += loss

        avg_acc = acc_total / len(self.feed_dict_test_list)
        avg_loss = loss_total / len(self.feed_dict_test_list)

        summary = tf.Summary()
        summary.value.add(tag='accuracy', simple_value=avg_acc)
        summary.value.add(tag='loss', simple_value=avg_loss)
        summary_writer.add_summary(summary, steps)

        end = time.time()

        return avg_acc, avg_loss, end - start

    def build_feed_dict(self, root):
        node_list = []
        tree_util.depth_first_traverse(root, node_list, lambda node, node_list: node_list.append(node))
        node_to_index = helper.reverse_dict(node_list)
        # TODO der er sikkert noget galt her

        feed_dict = {
            self.is_leaf_array: [node.is_leaf for node in node_list],
            self.word_index_array: [self.data.word_embed_util.get_idx(node.value) for node in node_list],
            # todo måske wrap på en anden måde
            self.left_child_array: [node_to_index[node.left_child] if node.left_child is not None else -1 for node in
                                    node_list],
            self.right_child_array: [node_to_index[node.right_child] if node.right_child is not None else -1 for node in
                                     node_list],
            self.label_array: [node.label for node in node_list]
        }

        return feed_dict

    def run_tensorboard(self):
        if FLAGS.run_tensorboard:
            os.system(
                'tensorboard --logdir=/home/dzach/Documents/Aarhus\ Universitet/Speciale/code/fraud_detector/logs/rnn &')
            webbrowser.open('http://0.0.0.0:6006/')


import atexit
import shutil


@atexit.register
def clean_tensorboard():
    # if FLAGS.run_tensorboard:
    shutil.rmtree(FLAGS.logs_dir + 'rnn')
    os.mkdir(FLAGS.logs_dir + 'rnn')
    os.mkdir(FLAGS.logs_dir + 'rnn/test')
    os.mkdir(FLAGS.logs_dir + 'rnn/train')
    os.system('fuser -k 6006/tcp')
