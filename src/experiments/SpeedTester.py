from models.treeRNN import treeRNN
from models.treeRNN_batch import treeRNN_batch
from models.treeRNN_neerbek import treeRNN_neerbek
from utils import helper, data_util
from trainers import TreeTrainer as trainer
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os


def make_experiment_folder():
    if not os.path.exists("../experiments/"):
        os.mkdir("../experiments/")


def run1(models=[treeRNN, treeRNN_batch],
         batch_sizes=[2 ** i for i in range(1, 10)],
         configs=[tf.ConfigProto(device_count={'GPU': 0}), None]):
    epochs = 3

    make_experiment_folder()

    _data_util = data_util.DataUtil()
    data = _data_util.get_data()

    run_times_list = []
    epoch_times_list = []
    for model in models:
        for config in configs:
            avg_run_times = []
            avg_epoch_times = []
            for batch_size in batch_sizes:
                run_times = []
                epoch_times = []
                with tf.Graph().as_default():
                    trainer.train(model(data, "test/"), load=False, config=config, batch_size=batch_size, epochs=epochs,
                                  run_times=run_times, epoch_times=epoch_times)
                avg_run_times.append(np.average(run_times))
                avg_epoch_times.append(np.average(epoch_times))
            run_times_list.append(avg_run_times)
            epoch_times_list.append(avg_epoch_times)

    np.savez("../experiments/run1.npz", run_times_list=run_times_list, epoch_times_list=epoch_times_list)


def plot1(batch_sizes=[2 ** i for i in range(1, 10)], labels=["Neerbek - CPU", "Neerbek - GPU", "Our - CPU", "Our - GPU"]):
    tmp = np.load("../experiments/SpeedTester.npz")

    run_times_list = tmp["run_times_list"].tolist()
    epoch_times_list = tmp["epoch_times_list"].tolist()

    plt.clf()
    for avg_epoch_times, label in zip(epoch_times_list, labels):
        plt.plot(batch_sizes, np.array(avg_epoch_times) / 60, label=label)
    plt.legend()
    plt.xlabel('batch size')
    plt.ylabel('minutes')
    plt.show()

    plt.clf()
    for avg_run_times, label in zip(run_times_list, labels):
        plt.plot(batch_sizes, np.array(avg_run_times) / 60, label=label)
    plt.legend()
    plt.xlabel('batch size')
    plt.ylabel('minutes')
    plt.show()

    plt.clf()


def run2():
    make_experiment_folder()
    epochs = 3
    batch_sizes = [2 ** i for i in range(1, 10)]
    config_CPU = tf.ConfigProto(device_count={'GPU': 0})
    config_GPU = None

    to_be_tested = [
        (treeRNN, config_CPU),
        (treeRNN_batch, config_GPU),
        (treeRNN_neerbek, config_GPU)
    ]

    _data_util = data_util.DataUtil()
    data = _data_util.get_data()

    run_times_list = []
    epoch_times_list = []
    for model, config in to_be_tested:
        avg_run_times = []
        avg_epoch_times = []
        for batch_size in batch_sizes:
            run_times = []
            epoch_times = []
            with tf.Graph().as_default():
                trainer.train(model(data, "test/"), load=False, config=config, batch_size=batch_size, epochs=epochs,
                              run_times=run_times, epoch_times=epoch_times)
            avg_run_times.append(np.average(run_times))
            avg_epoch_times.append(np.average(epoch_times))
        run_times_list.append(avg_run_times)
        epoch_times_list.append(avg_epoch_times)


    np.savez("../experiments/run2.npz", run_times_list=run_times_list, epoch_times_list=epoch_times_list)
