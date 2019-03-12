from utils import helper, data_util
from trainers import TreeTrainer as trainer
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def run(models, batch_sizes, configs):
    epochs = 3

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

    np.savez("SpeedTester.npz", run_times_list=run_times_list, epoch_times_list=epoch_times_list)


def plot(batch_sizes, labels):
    tmp = np.load("SpeedTester.npz")

    run_times_list = tmp["run_times_list"].tolist()
    epoch_times_list = tmp["epoch_times_list"].tolist()

    for avg_epoch_times, label in zip(epoch_times_list, labels):
        plt.plot(batch_sizes, avg_epoch_times, label=label)
    plt.show()

    for avg_run_times, label in zip(run_times_list, labels):
        plt.plot(batch_sizes, avg_run_times, label=label)
    plt.show()
