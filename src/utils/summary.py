import utils.helper as helper
from utils.flags import FLAGS
import os
import shutil
import tensorflow as tf
import numpy as np


class summarizer():
    TRAIN = "train"
    VAL = "validation"
    TEST = "test"
    all_data_sets = [TRAIN, VAL, TEST]
    writer = {TRAIN: None, VAL: None, TEST: None}
    rounds = {TRAIN: 0, VAL: 0, TEST: 0}
    acc = {TRAIN: 0, VAL: 0, TEST: 0}
    loss = {TRAIN: 0, VAL: 0, TEST: 0}
    history = {TRAIN: [], VAL: [], TEST: []}
    best_acc = {TRAIN: 0, VAL: 0, TEST: 0}
    _new_best = {TRAIN: False, VAL: False, TEST: False}

    def __init__(self, model_name, sess):
        self.model_name = model_name
        self.sess = sess


    def construct_writers(self):
        directory = FLAGS.logs_dir + self.model_name
        self.writer[self.TRAIN] = tf.summary.FileWriter(directory + self.TRAIN, self.sess.graph)
        self.writer[self.VAL] = tf.summary.FileWriter(directory + self.VAL)
        self.writer[self.TEST] = tf.summary.FileWriter(directory + self.TEST)

    def load(self):
        tmp = np.load(FLAGS.histories_dir + self.model_name + 'history.npz')

        for data_set in self.all_data_sets:
            self.history[data_set] = tmp[data_set].tolist()
            self.best_acc[data_set] = np.max([acc for epoch, acc, loss in self.history[data_set]])


        self.construct_writers()

    def initialize(self):
        self.construct_dir()
        self.construct_writers()

    def construct_dir(self):
        helper._print("Constructing directories...")

        directory = FLAGS.logs_dir + self.model_name
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.mkdir(directory)
        os.mkdir(directory + self.TRAIN)
        os.mkdir(directory + self.VAL)
        os.mkdir(directory + self.TEST)
        if not os.path.exists(FLAGS.histories_dir):
            os.mkdir(FLAGS.histories_dir)
        if not os.path.exists(FLAGS.histories_dir + self.model_name):
            os.mkdir(FLAGS.histories_dir + self.model_name)

        helper._print("Directories constructed!")

    def add(self, data_set, acc, loss):
        self.rounds[data_set] += 1
        self.acc[data_set] += acc
        self.loss[data_set] += loss

    def write_and_reset(self, data_set, epoch, _print=False):
        avg_loss = self.loss[data_set] / self.rounds[data_set]
        avg_acc = self.acc[data_set] / self.rounds[data_set]
        self.rounds[data_set] = 0
        self.acc[data_set] = 0
        self.loss[data_set] = 0
        self.history[data_set].append((epoch,
                                       avg_acc,
                                       avg_loss))

        self.write_to_summary(data_set, avg_acc, avg_loss, epoch)

        if avg_acc > self.best_acc[data_set]:
            self.best_acc[data_set] = avg_acc
            self._new_best[data_set] = True
        else:
            self._new_best[data_set] = False

        if _print:
            helper._print(data_set.capitalize(), "-", "acc:", avg_acc, "loss:", avg_loss)

    def write_to_summary(self, data_set, acc, loss, epoch):
        summary = tf.Summary()
        summary.value.add(tag='accuracy', simple_value=acc)
        summary.value.add(tag='loss', simple_value=loss)
        self.writer[data_set].add_summary(summary, epoch)

    def compute(self, data_set, data, model, sess, epoch, _print=False):
        # for step, batch in enumerate(helper.batches(data, FLAGS.batch_size)):
        #     feed_dict = model.build_feed_dict(batch)
        #     acc, loss = sess.run([model.acc, model.loss], feed_dict=feed_dict)
        #     self.add(data_set, acc, loss)

        feed_dict = model.build_feed_dict(data)
        acc, loss = sess.run([model.acc, model.loss], feed_dict=feed_dict)
        self.add(data_set, acc, loss)
        self.write_and_reset(data_set, epoch, _print=_print)

    def save_all(self):
        np.savez(FLAGS.histories_dir + self.model_name + 'history.npz',
                 train=self.history[self.TRAIN],
                 test=self.history[self.TEST],
                 validation=self.history[self.VAL])

    def new_best(self, data_set):
        return self._new_best[data_set]

    def close(self):
        self.writer[self.TRAIN].close()
        self.writer[self.VAL].close()
        self.writer[self.TEST].close()