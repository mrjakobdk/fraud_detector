import csv
import sys
import math
from time import time

import matplotlib.pyplot as plt
import utils.helper as helper
import os
import shutil
import tensorflow as tf
import numpy as np

from utils import directories, constants
from utils.flags import FLAGS
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
    pre_best_acc = {TRAIN: 0, VAL: 0, TEST: 0}
    best_loss = {TRAIN: math.inf, VAL: math.inf, TEST: math.inf}
    _new_best_acc = {TRAIN: False, VAL: False, TEST: False}
    _new_best_loss = {TRAIN: False, VAL: False, TEST: False}
    delta_time = 0

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
        "batch": 0,
        "best_batch": 0,
        "best_epoch": 0,
        "epoch": 0,
        "best_time": 0,
        "total_time": 0,
        "dropping_count": 0,
        "converging_count": 0,
        "dropping_acc": 0,
        "converging_acc": 0,
        "pre_epoch": 0,
        "pre_batch": 0,
        "main_count": 0,
    }

    performance_train = {
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

    performance_val = {
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

    performance_test = {
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
        self.time_start = time()
        self.speed = {
            "batch": 0,
            "best_batch": 0,
            "best_epoch": 0,
            "epoch": 0,
            "best_time": 0,
            "total_time": 0,
            "dropping_count": 0,
            "converging_count": 0,
            "dropping_acc": 0,
            "converging_acc": 0,
            "pre_epoch": 0,
            "pre_batch": 0,
            "main_count": 0,
        }

    def construct_writers(self):
        self.writer[self.TRAIN] = tf.summary.FileWriter(directories.LOGS_TRAIN_DIR(self.model_name))
        self.writer[self.VAL] = tf.summary.FileWriter(directories.LOGS_VAL_DIR(self.model_name))
        self.writer[self.TEST] = tf.summary.FileWriter(directories.LOGS_TEST_DIR(self.model_name))

    def load(self):

        if os.path.exists(directories.PARAMETERS_FILE(self.model_name)):
            self.parameters = helper.load_dict(directories.PARAMETERS_FILE(self.model_name))
        if os.path.exists(directories.PERFORMANCE_TEST_FILE(self.model_name)):
            self.performance_test = helper.load_dict(directories.PERFORMANCE_TEST_FILE(self.model_name))
        if os.path.exists(directories.PERFORMANCE_VAL_FILE(self.model_name)):
            self.performance_val = helper.load_dict(directories.PERFORMANCE_VAL_FILE(self.model_name))
        if os.path.exists(directories.PERFORMANCE_TRAIN_FILE(self.model_name)):
            self.performance_train = helper.load_dict(directories.PERFORMANCE_TRAIN_FILE(self.model_name))
        if os.path.exists(directories.SPEED_FILE(self.model_name)):
            self.speed = helper.load_dict(directories.SPEED_FILE(self.model_name))
        if os.path.exists(directories.BEST_ACC_FILE(self.model_name)):
            self.best_acc = helper.load_dict(directories.BEST_ACC_FILE(self.model_name))
        if os.path.exists(directories.BEST_LOSS_FILE(self.model_name)):
            self.best_loss = helper.load_dict(directories.BEST_LOSS_FILE(self.model_name))

        if os.path.exists(directories.HISTORIES_FILE(self.model_name)):
            tmp = np.load(directories.HISTORIES_FILE(self.model_name))
            for data_set in self.all_data_sets:
                self.history[data_set] = tmp[data_set].tolist()
        self.construct_writers()

    def initialize(self):
        self.construct_writers()

    def construct_dir(self):
        model_name = self.model_name
        helper._print("Constructing directories...")

        if not os.path.exists(directories.TRAINED_MODELS_DIR):
            os.mkdir(directories.TRAINED_MODELS_DIR)

        if FLAGS.load_model:
            if not os.path.exists(directories.TMP_MODEL_DIR(model_name)):
                self.make_model_dirs(model_name)
        else:
            if os.path.exists(directories.MODEL_DIR(model_name)):
                shutil.rmtree(directories.MODEL_DIR(model_name))
            self.make_model_dirs(model_name)

        helper._print("Directories constructed!")

    def make_model_dirs(self, model_name):
        if not os.path.exists(directories.MODEL_DIR(model_name)):
            os.mkdir(directories.MODEL_DIR(model_name))
        os.mkdir(directories.LOGS_DIR(model_name))
        os.mkdir(directories.LOGS_TRAIN_DIR(model_name))
        os.mkdir(directories.LOGS_VAL_DIR(model_name))
        os.mkdir(directories.LOGS_TEST_DIR(model_name))
        os.mkdir(directories.HISTORIES_DIR(model_name))
        os.mkdir(directories.BEST_MODEL_DIR(model_name))
        for data_set in self.all_data_sets:
            os.mkdir(directories.BEST_MODEL_DIR(model_name, data_set))
        os.mkdir(directories.PLOTS_DIR(model_name))

    def add(self, data_set, acc, loss):
        self.rounds[data_set] += 1
        self.acc[data_set] += acc
        self.loss[data_set] += loss

    def write_and_reset(self, data_set, _print=False):
        avg_loss = self.loss[data_set] / self.rounds[data_set]
        avg_acc = self.acc[data_set] / self.rounds[data_set]
        self.rounds[data_set] = 0
        self.acc[data_set] = 0
        self.loss[data_set] = 0
        self.history[data_set].append((self.speed["epoch"],
                                       avg_acc,
                                       avg_loss))
        if math.isnan(avg_loss):
            return False
        self.write_to_summary(data_set, avg_acc, avg_loss, self.speed["epoch"])

        if avg_acc >= self.best_acc[data_set]:
            self.best_acc[data_set] = avg_acc
            self._new_best_acc[data_set] = True
            helper.save_dict(self.best_acc, placement=directories.BEST_ACC_FILE(self.model_name))
        else:
            self._new_best_acc[data_set] = False

        if avg_loss <= self.best_loss[data_set]:
            self.best_loss[data_set] = avg_loss
            self._new_best_loss[data_set] = True
            helper.save_dict(self.best_loss, placement=directories.BEST_LOSS_FILE(self.model_name))
        else:
            self._new_best_loss[data_set] = False

        if _print:
            helper._print(data_set.capitalize(), "-", "acc:", avg_acc, "loss:", avg_loss)
        return True

    def write_to_summary(self, data_set, acc, loss, epoch):
        summary = tf.Summary()
        summary.value.add(tag='accuracy', simple_value=acc)
        summary.value.add(tag='loss', simple_value=loss)
        self.writer[data_set].add_summary(summary, epoch)

    def compute(self, data_set, data, model, _print=False):
        if FLAGS.use_gpu:
            feed_dict, _ = model.build_feed_dict(data)
            acc, loss = self.sess.run([model.acc, model.loss], feed_dict=feed_dict)
            self.add(data_set, acc, loss)
            self.write_and_reset(data_set, _print=_print)
        else:
            for batch in helper.batches(data, 2):
                feed_dict, _ = model.build_feed_dict(batch)
                acc, loss = self.sess.run([model.acc, model.loss], feed_dict=feed_dict)
                self.add(data_set, acc, loss)
            self.write_and_reset(data_set, _print=_print)


    def save_history(self):
        np.savez(directories.HISTORIES_FILE(self.model_name),
                 train=self.history[self.TRAIN],
                 test=self.history[self.TEST],
                 validation=self.history[self.VAL])

    def save_performance(self, model):
        data = model.data
        p = Performance(data.test_trees, model, self.sess)
        self.performance_test = p.get_performance()
        p.plot_ROC(placement=directories.ROC_TEST_PLOT(self.model_name))
        helper.save_dict(self.performance_test, placement=directories.PERFORMANCE_TEST_FILE(self.model_name))

        p = Performance(data.val_trees, model, self.sess)
        self.performance_val = p.get_performance()
        p.plot_ROC(placement=directories.ROC_VAL_PLOT(self.model_name))
        helper.save_dict(self.performance_val, placement=directories.PERFORMANCE_VAL_FILE(self.model_name))

        p = Performance(data.train_trees, model, self.sess)
        self.performance_train = p.get_performance()
        p.plot_ROC(placement=directories.ROC_TRAIN_PLOT(self.model_name))
        helper.save_dict(self.performance_train, placement=directories.PERFORMANCE_TRAIN_FILE(self.model_name))

        self.plot_history()

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

        with open(directories.SYS_ARG_FILE(self.model_name), "a") as text_file:
            text_file.write(str(sys.argv) + "\n")
            text_file.write(str(FLAGS) + "\n")

    def save_speed(self):
        # todo keep track of total time
        helper.save_dict(self.speed, placement=directories.SPEED_FILE(self.model_name))

    def new_best_acc(self, data_set):
        if self._new_best_acc[data_set]:
            self.speed["best_epoch"] = self.speed["epoch"]
            self.speed["best_batch"] = self.speed["batch"]
            self.speed["best_time"] = self.speed["total_time"]

        return self._new_best_acc[data_set]

    # def new_best_loss(self, data_set):
    #     return self._new_best_loss[data_set]

    def close(self):
        self.writer[self.TRAIN].close()
        self.writer[self.VAL].close()
        self.writer[self.TEST].close()

    def plot_history(self):
        plt.clf()
        epochs = len(self.history[self.TEST])
        for data_set in self.all_data_sets:
            acc = []
            for i in range(epochs):
                acc.append(self.history[data_set][i][1])
            plt.plot(list(range(1, epochs + 1)), acc, label=data_set)
        plt.legend()
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.savefig(directories.ACC_HISTORY_PLOT(self.model_name))

        plt.clf()
        epochs = len(self.history[self.TEST])
        for data_set in self.all_data_sets:
            acc = []
            for i in range(epochs):
                acc.append(self.history[data_set][i][2])
            plt.plot(list(range(1, epochs + 1)), acc, label=data_set)
        plt.legend()
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.savefig(directories.LOSS_HISTORY_PLOT(self.model_name))

    def print_performance(self):
        helper._print_header("Final stats for best model")

        helper._print("Best epoch:", self.speed["best_epoch"])
        helper._print("Total running time:",
                      str(int(self.speed["best_time"] / (60 * 60))) + "h",
                      str((int(self.speed["best_time"] / 60) % 60)) + "m")
        helper._print("Total epochs:", self.speed["epoch"])
        helper._print("Total running time:",
                      str(int(self.speed["total_time"] / (60 * 60))) + "h",
                      str((int(self.speed["total_time"] / 60) % 60)) + "m")

        helper._print("Test:", self.performance_test)
        helper._print("Val:", self.performance_val)
        helper._print("train:", self.performance_train)

    def epoch_inc(self):
        self.speed["epoch"] += 1

    def batch_inc(self):
        self.speed["batch"] += 1

    def get_epoch(self):
        return self.speed["epoch"]

    def dropping(self):
        return self.best_acc[self.TRAIN] >= FLAGS.pretrain_max_acc or self.speed[
            "epoch"] >= FLAGS.pretrain_max_epoch

    # def dropping_tick(self):
    #     if self.best_acc[self.TRAIN] - self.speed["dropping_acc"] > FLAGS.acc_min_delta_drop:
    #         self.speed["dropping_acc"] = self.best_acc[self.TRAIN]
    #         self.speed["dropping_count"] = 0
    #     else:
    #         self.speed["dropping_count"] += 1
    #     helper._print(
    #         f"Dropping for {self.speed['dropping_count']}/{FLAGS.pretrain_stop_count} epochs. Prev best train acc: {self.best_acc[self.TRAIN]}")

    def converging(self):
        return self.speed["converging_count"] >= FLAGS.conv_cond

    def converging_tick(self):
        if self.best_acc[self.VAL] - self.speed["converging_acc"] > FLAGS.acc_min_delta_conv:
            self.speed["converging_acc"] = self.best_acc[self.VAL]
            self.speed["converging_count"] = 0
        else:
            self.speed["converging_count"] += 1
        helper._print(
            f"Converging in {self.speed['converging_count']}/{FLAGS.conv_cond} epochs. Prev best val acc: {self.speed['converging_acc']}")

    def time_tick(self, msg="Epoch time:"):
        time_end = time()
        self.delta_time = time_end - self.time_start
        helper._print(msg, str(int(self.delta_time / 60)) + "m " + str(int(self.delta_time % 60)) + "s")
        self.speed["total_time"] += self.delta_time
        self.time_start = time_end

    def at_max_epoch(self):
        return self.parameters["max_epochs"] != 0 and self.speed["epoch"] >= self.parameters["max_epochs"]

    def get_time(self):
        return self.delta_time

    def interrupt(self):
        stop = os.path.exists("stop.please")
        if stop:
            helper._print("!!!INTERRUPT!!!")
        return stop

    def pre_tick(self):
        self.speed["pre_epoch"] = self.speed["epoch"]
        self.speed["pre_batch"] = self.speed["batch"]

    def main_count_tick(self):
        self.speed["main_count"] += 1

    def re_cluster(self):
        return self.speed["main_count"] == 1 or (FLAGS.use_multi_cluster and self.speed["main_count"] % int(FLAGS.pretrain_max_epoch/4)==0)

    def load_cluster_predictions(self):
        return np.load(directories.CLUSTER_FILE(self.model_name))

    def save_cluster_predictions(self, cluster_predictions):
        np.save(directories.CLUSTER_FILE(self.model_name), cluster_predictions)
