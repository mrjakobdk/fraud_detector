import csv
import sys
import math

import utils.helper as helper
from utils import directories
from utils.flags import FLAGS
import os
import shutil
import tensorflow as tf
import numpy as np

from utils.performance import Performance


class summarizer():
    # keeping track of history
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
    best_loss = {TRAIN: math.inf, VAL: math.inf, TEST: math.inf}
    _new_best_acc = {TRAIN: False, VAL: False, TEST: False}
    _new_best_loss = {TRAIN: False, VAL: False, TEST: False}

    #########
    parameters = {
        "lr": 0,
        "lr_end": 0,
        "gpu": False,
        "lr_decay": 0,
        "conv_cond": 0,
        "model": "",
        "number_variables": 0,
        "max_epochs": 0,
    }
    # summary at the time of best performance
    speed = {
        "epochs": 0,
        "total_time": 0,
    }
    performance = {
        "accuracy": 0,
        "auc": 0,
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "precision": 0,
        "recall": 0,
        "f1": 0,
    }

    def __init__(self, model_name, sess):
        self.model_name = model_name
        self.sess = sess

    def construct_writers(self):
        self.writer[self.TRAIN] = tf.summary.FileWriter(directories.LOGS_TRAIN_DIR(self.model_name))
        self.writer[self.VAL] = tf.summary.FileWriter(directories.LOGS_VAL_DIR(self.model_name))
        self.writer[self.TEST] = tf.summary.FileWriter(directories.LOGS_TEST_DIR(self.model_name))

    def load(self):
        tmp = np.load(directories.HISTORIES_FILE(self.model_name))

        if os.path.exists(directories.PARAMETERS_FILE(self.model_name)):
            self.parameters = helper.load_dict(directories.PARAMETERS_FILE(self.model_name))
        if os.path.exists(directories.PERFORMANCE_FILE(self.model_name)):
            self.performance = helper.load_dict(directories.PERFORMANCE_FILE(self.model_name))
        if os.path.exists(directories.SPEED_FILE(self.model_name)):
            self.speed = helper.load_dict(directories.SPEED_FILE(self.model_name))

        for data_set in self.all_data_sets:
            self.history[data_set] = tmp[data_set].tolist()
            self.best_acc[data_set] = np.max([acc for epoch, acc, loss in self.history[data_set]])
        self.construct_writers()

    def initialize(self):
        self.construct_writers()

    def construct_dir(self):
        model_name = self.model_name
        helper._print("Constructing directories...")

        if not os.path.exists(directories.TRAINED_MODELS_DIR):
            os.mkdir(directories.TRAINED_MODELS_DIR)

        if FLAGS.load_model:
            if not os.path.exists(directories.MODEL_DIR(model_name)):
                helper._print("!!! No model named:", model_name, "!!!")
                sys.exit()
        else:
            if os.path.exists(directories.MODEL_DIR(model_name)):
                shutil.rmtree(directories.MODEL_DIR(model_name))
            os.mkdir(directories.MODEL_DIR(model_name))
            os.mkdir(directories.LOGS_DIR(model_name))
            os.mkdir(directories.LOGS_TRAIN_DIR(model_name))
            os.mkdir(directories.LOGS_VAL_DIR(model_name))
            os.mkdir(directories.LOGS_TEST_DIR(model_name))
            os.mkdir(directories.HISTORIES_DIR(model_name))
            os.mkdir(directories.BEST_MODEL_DIR(model_name))
            os.mkdir(directories.PLOTS_DIR(model_name))

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
            self._new_best_acc[data_set] = True
        else:
            self._new_best_acc[data_set] = False

        if avg_loss < self.best_loss[data_set]:
            self.best_loss[data_set] = avg_loss
            self._new_best_loss[data_set] = True
        else:
            self._new_best_loss[data_set] = False

        if _print:
            helper._print(data_set.capitalize(), "-", "acc:", avg_acc, "loss:", avg_loss)

    def write_to_summary(self, data_set, acc, loss, epoch):
        summary = tf.Summary()
        summary.value.add(tag='accuracy', simple_value=acc)
        summary.value.add(tag='loss', simple_value=loss)
        self.writer[data_set].add_summary(summary, epoch)

    def compute(self, data_set, data, model, epoch, _print=False):
        feed_dict = model.build_feed_dict(data)
        acc, loss = self.sess.run([model.acc, model.loss], feed_dict=feed_dict)
        self.add(data_set, acc, loss)
        self.write_and_reset(data_set, epoch, _print=_print)

    def save_history(self, epoch_times, run_times):
        np.savez(directories.HISTORIES_FILE(self.model_name),
                 train=self.history[self.TRAIN],
                 test=self.history[self.TEST],
                 validation=self.history[self.VAL],
                 epoch_times=epoch_times,
                 run_times=run_times)

    def save_performance(self, data, model):
        p = Performance(data, model, self.sess)
        self.performance = p.get_performance()
        p.plot_ROC(placement=directories.ROC_PLOT(self.model_name))
        self.plot_history()
        helper.save_dict(self.performance, placement=directories.PERFORMANCE_FILE(self.model_name))

    def save_parameters(self, lr, lr_end, gpu, lr_decay, conv_cond, model, number_variables,
                        max_epochs, optimizer):
        self.parameters = {
            "lr": lr,
            "lr_end": lr_end,
            "gpu": gpu,
            "lr_decay": lr_decay,
            "conv_cond": conv_cond,
            "model": model,
            "number_variables": number_variables,
            "max_epochs": max_epochs,
            "optimizer": optimizer,
            "data_set": "enron",  # todo should not be default
        }
        helper.save_dict(self.parameters, placement=directories.PARAMETERS_FILE(self.model_name))

        with open(directories.SYS_ARG_FILE(self.model_name), "w") as text_file:
            text_file.write(str(sys.argv))

    def save_speed(self, epochs, total_time):
        self.speed = {
            "epochs": epochs,
            "total_time": total_time,
        }
        helper.save_dict(self.speed, placement=directories.SPEED_FILE(self.model_name))

    def new_best_acc(self, data_set):
        return self._new_best_acc[data_set]

    def new_best_loss(self, data_set):
        return self._new_best_loss[data_set]

    def close(self):
        self.writer[self.TRAIN].close()
        self.writer[self.VAL].close()
        self.writer[self.TEST].close()

    def plot_history(self):
        pass
