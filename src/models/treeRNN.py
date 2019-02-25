import tensorflow as tf
from utils.flags import FLAGS
import utils.helper as helper
import utils.tree_util as tree_util
import numpy as np
import sys
import os
import shutil
import webbrowser
import time


class tRNN:
    feed_dict_list = {"val": [], "test": [], "train": []}

    def __init__(self, data):
        """
        :param data: utils.data
        """
        helper._print_header("Constructing tRNN constants, placeholders and variables")

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
        W_l = tf.get_variable(name='W_l', shape=[FLAGS.sentence_embedding_size, FLAGS.word_embedding_size],
                              initializer=weight_initializer)
        W_r = tf.get_variable(name='W_r', shape=[FLAGS.sentence_embedding_size, FLAGS.word_embedding_size],
                              initializer=weight_initializer)
        self.W_l = W_l
        self.W_r = W_r

        # phrase weights
        U_l = tf.get_variable(name='U_l', shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                              initializer=weight_initializer)
        U_r = tf.get_variable(name='U_r', shape=[FLAGS.sentence_embedding_size, FLAGS.sentence_embedding_size],
                              initializer=weight_initializer)
        self.U_l = U_l
        self.U_r = U_r

        # bias
        b_W = tf.get_variable(name='b_W', shape=[FLAGS.sentence_embedding_size, 1], initializer=bias_initializer)
        b_U = tf.get_variable(name='b_U', shape=[FLAGS.sentence_embedding_size, 1], initializer=bias_initializer)

        # classifier weights
        V = tf.get_variable(name='V', shape=[FLAGS.label_size, FLAGS.sentence_embedding_size],
                            initializer=xavier_initializer)
        b_p = tf.get_variable(name='b_p', shape=[FLAGS.label_size, 1], initializer=bias_initializer)
        self.V = V
        self.b_p = b_p

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
            self.learning_rate = FLAGS.learning_rate

        if FLAGS.optimizer == "adam":
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)
        else:
            self.train_op = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss,
                                                                                   global_step=self.global_step)
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
        #todo change to fit batch
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

    def get_loss_root(self):
        root_index = self.o_array.size() - 1
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=tf.reshape(self.o_array.read(root_index), [-1, FLAGS.label_size]),
                labels=tf.gather(self.label_array, root_index)))
        return loss

    def get_loss(self):
        # todo change to the correct loss
        # Loss
        # pro_1 = tf.matmul(self.weight,
        #                   tf.matmul(self.label_array,
        #                             tf.log(self.o_array.concat())))
        # pro_2 = tf.matmul(1 - self.label_array, tf.log(1 - self.o_array.concat()))
        # loss = - tf.reduce_sum(pro_1 + pro_2)
        # tf.summary.scalar("loss", loss)
        # Todo: check reshape.

        # weights = tf.constant([0.6646, 1 / 0.6646])
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
        helper._print_header("Training tRNN")
        helper._print("Test ration:", tree_util.ratio_of_labels(self.data.test_trees))
        helper._print("Validation ration:", tree_util.ratio_of_labels(self.data.val_trees))
        helper._print("Train ration:", tree_util.ratio_of_labels(self.data.train_trees))

        # todo make a flag for this
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )

        with tf.Session(config=config) as sess:
            model_placement = FLAGS.models_dir + FLAGS.model_name + "model.ckpt"

            # Summary writer for both the training and the set acc and loss - used for tensorboard
            self.make_needed_dir()
            directory = FLAGS.logs_dir + FLAGS.model_name
            train_writer = tf.summary.FileWriter(directory + 'train', sess.graph)
            validation_writer = tf.summary.FileWriter(directory + 'validation')
            test_writer = tf.summary.FileWriter(directory + 'test')

            history = self.get_history()
            starting_steps = 0
            best_acc = 0

            # Run the init
            saver = tf.train.Saver()
            self.run_tensorboard()
            if FLAGS.load_model:
                history, starting_steps, best_acc = self.load_history()
                helper._print("Previously", starting_steps, "steps has been ran, best acc was:", best_acc)

                self.load_model(sess, model_placement, saver)
                self.write_history_to_summary(history, train_writer, validation_writer, test_writer)
                sess.run(tf.assign(self.global_step, starting_steps))
            else:
                sess.run(self.init)
                self.handle_val_test(history, sess, test_writer, 0, validation_writer)

            start_time = time.time()
            print_interval = FLAGS.print_step_interval / FLAGS.batch_size
            loss_total = 0
            acc_total = 0
            for epoch in range(FLAGS.epochs):
                helper._print_header("Epoch " + str(epoch + 1))

                for step, tree in enumerate(helper.batches(np.random.permutation(self.data.train_trees),
                                                           FLAGS.batch_size)):  # todo build train get_trees
                    if step % int(print_interval) == 0:
                        total_step = starting_steps + epoch * int(len(self.data.train_trees)/FLAGS.batch_size) + step
                        helper._print("Step:", total_step)
                        helper._print("Learning rate:", sess.run(self.learning_rate))

                        avg_acc = acc_total / print_interval
                        avg_loss = loss_total / print_interval
                        if epoch != 0 or step != 0:
                            self.write_to_summary(avg_acc, avg_loss, total_step, train_writer)
                            helper._print("Train -  acc:", avg_acc, "loss:", avg_loss)
                            history["train"].append((total_step, avg_acc, avg_loss))

                            val_acc = self.handle_val_test(history, sess, test_writer, total_step,
                                                           validation_writer)

                            loss_total = 0
                            acc_total = 0

                            if val_acc > best_acc:
                                best_acc = val_acc
                                helper._print("A better model was found!")

                                saver.save(sess, model_placement)

                                np.savez(FLAGS.histories_dir + FLAGS.model_name + 'history.npz',
                                         train=history["train"],
                                         test=history["test"],
                                         val=history["val"],
                                         total_steps=total_step,
                                         best_acc=best_acc)

                                helper._print("Model saved!")

                    feed_dict = self.build_feed_dict_batch(tree)  # todo maybe change to batches
                    _, acc, loss = sess.run([self.train_op, self.acc, self.loss],
                                            feed_dict=feed_dict)

                    acc_total += acc
                    loss_total += loss

                helper._print("Avg Epoch Time:", (time.time() - start_time) / (epoch + 1) / 60, "m")

    def write_to_summary(self, acc, loss, steps, writer):
        summary = tf.Summary()
        summary.value.add(tag='accuracy', simple_value=acc)
        summary.value.add(tag='loss', simple_value=loss)
        writer.add_summary(summary, steps)

    def handle_val_test(self, history, sess, test_writer, total_step, validation_writer):
        val_acc, val_loss, val_time = self.compute_acc_loss(self.data.val_trees, sess,
                                                            validation_writer,
                                                            total_step)
        helper._print("Validation -  acc:", val_acc, "loss:", val_loss, "time:", val_time)
        history["val"].append((total_step, val_acc, val_loss))
        test_acc, test_loss, test_time = self.compute_acc_loss(self.data.test_trees, sess,
                                                               test_writer,
                                                               total_step, data_set="test")
        helper._print("Test -  acc:", test_acc, "loss:", test_loss, "time:", test_time)
        history["test"].append((total_step, test_acc, test_loss))
        return val_acc

    def load_history(self):
        tmp = np.load(FLAGS.histories_dir + FLAGS.model_name + 'history.npz')
        history = self.get_history()
        history["train"] = tmp["train"].tolist()
        history["test"] = tmp["test"].tolist()
        history["val"] = tmp["val"].tolist()
        return history, tmp["total_steps"], tmp["best_acc"]

    def write_history_to_summary(self, history, train_writer, validation_writer, test_writer):
        helper._print("Restoring summary...")

        def write_history(point_list, writer):
            for point in point_list:
                steps, acc, loss = point
                self.write_to_summary(acc, loss, steps, writer)

        write_history(history["train"], train_writer)
        write_history(history["val"], validation_writer)
        write_history(history["test"], test_writer)

        helper._print("Summary restored!")

    def load_model(self, sess, model_placement, saver):
        helper._print("Restoring model...")
        saver.restore(sess, model_placement)
        helper._print("Model restored!")

    def make_needed_dir(self):
        helper._print("Constructing directories...")

        directory = FLAGS.logs_dir + FLAGS.model_name
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.mkdir(directory)
        os.mkdir(directory + 'train')
        os.mkdir(directory + 'validation')
        os.mkdir(directory + 'test')
        if not os.path.exists(FLAGS.histories_dir + FLAGS.model_name):
            os.mkdir(FLAGS.histories_dir + FLAGS.model_name)

        helper._print("Directories constructed!")

    def get_history(self):
        return {"train": [], "val": [], "test": []}

    def compute_acc_loss(self, data, sess, summary_writer, steps, data_set="val"):  # todo should only be for test
        start = time.time()
        loss_total = 0
        acc_total = 0

        if len(self.feed_dict_list[data_set]) == 0:
            for tree in data:
                self.feed_dict_list[data_set].append(self.build_feed_dict(tree))
        for step, feed_dict in enumerate(self.feed_dict_list[data_set]):
            acc, loss = sess.run([self.acc, self.loss],
                                 feed_dict=feed_dict)  # todo fix summary
            acc_total += acc
            loss_total += loss

        avg_acc = acc_total / len(self.feed_dict_list[data_set])
        avg_loss = loss_total / len(self.feed_dict_list[data_set])

        self.write_to_summary(avg_acc, avg_loss, steps, summary_writer)

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

    def build_feed_dict_batch(self, batch):
        node_list_list = []
        node_to_index_list = []
        for root in batch:
            node_list = []
            tree_util.depth_first_traverse(root, node_list, lambda node, node_list: node_list.append(node))
            node_list_list.append(node_list)
            node_to_index = helper.reverse_dict(node_list)
            node_to_index_list.append(node_to_index)
        # TODO der er sikkert noget galt her

        feed_dict = {
            self.is_leaf_array: helper.flatten([[node.is_leaf for node in node_list] for node_list in node_list_list]),
            self.word_index_array: helper.flatten([[self.data.word_embed_util.get_idx(node.value) for node in node_list]
                                                   for node_list in node_list_list]),
            # todo måske wrap på en anden måde
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

    def run_tensorboard(self):
        if FLAGS.run_tensorboard:
            os.system(
                'tensorboard --logdir=/home/dzach/Documents/Aarhus\ Universitet/Speciale/code/fraud_detector/logs/rnn &')
            webbrowser.open('http://0.0.0.0:6006/')


import atexit
import shutil


@atexit.register
def clean_tensorboard():
    if FLAGS.run_tensorboard:
        shutil.rmtree(FLAGS.logs_dir + FLAGS.model_name)
        os.mkdir(FLAGS.logs_dir + 'rnn')
        os.mkdir(FLAGS.logs_dir + 'rnn/test')
        os.mkdir(FLAGS.logs_dir + 'rnn/train')
        os.system('fuser -k 6006/tcp')
