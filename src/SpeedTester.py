import numpy as np
from models.trees.deepRNN import deepRNN
from models.trees.treeLSTM import treeLSTM
from models.trees.treeRNN import treeRNN
from models.trees.treeRNN_batch import treeRNN_batch
from models.trees.treeRNN_neerbek import treeRNN_neerbek
from models.trees.treeLSTM_tracker import treeLSTM_tracker
from models.words_embeddings.glove import GloVe
from utils import data_util, directories, constants
from trainers import TreeTrainer as trainer
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from utils.flags import FLAGS

def make_experiment_folder():
    if not os.path.exists("../experiments/"):
        os.mkdir("../experiments/")

def run_speed_test(placement):
    make_experiment_folder()
    epochs = 4
    batch_sizes = [4,16,32,64,128]
    config_CPU = False
    config_GPU = True

    to_be_tested = [
        (treeRNN_neerbek, config_GPU),
        (treeRNN_batch, config_GPU),
        (treeLSTM, config_GPU),
        (deepRNN, config_GPU),
        (treeLSTM_tracker, config_GPU),
        (treeRNN_neerbek, config_CPU),
        (treeRNN_batch, config_CPU),
        (treeLSTM, config_CPU),
        (deepRNN, config_CPU),
        (treeLSTM_tracker, config_CPU)
    ]

    labels = [
        "TreeRNN - GPU",
        "MTreeRNN - GPU",
        "TreeLSTM - GPU",
        "DeepRNN - GPU",
        "TreeLSTM w. Tracker - GPU",
        "TreeRNN - CPU",
        "MTreeRNN - CPU",
        "TreeLSTM - CPU",
        "DeepRNN - CPU",
        "TreeLSTM w. Tracker - CPU",
    ]

    _data_util = data_util.DataUtil()
    data = _data_util.get_data()

    word_embed = GloVe(mode=constants.PRETRAINED_MODE, dimensions=FLAGS.word_embedding_size)

    run_times_list = []
    epoch_times_list = []
    for model, config in to_be_tested:
        avg_run_times = []
        avg_epoch_times = []
        for batch_size in batch_sizes:
            run_times = []
            epoch_times = []
            with tf.Graph().as_default():
                trainer.train(model(data, word_embed, FLAGS.model_name), load=False, gpu=config,
                              batch_size=batch_size, epochs=epochs,
                              run_times=run_times, epoch_times=epoch_times, compute_performance=False)
            avg_run_times.append(np.average(run_times))
            avg_epoch_times.append(np.average(epoch_times))
        run_times_list.append(avg_run_times)
        epoch_times_list.append(avg_epoch_times)

    np.savez(placement, run_times_list=run_times_list, epoch_times_list=epoch_times_list, labels=labels,
             batch_sizes=batch_sizes)


def plot(placement, x_label='batch size', y_label='minutes'):
    tmp = np.load(placement)

    run_times_list = tmp["run_times_list"].tolist()
    epoch_times_list = tmp["epoch_times_list"].tolist()
    labels = tmp["labels"].tolist()
    batch_sizes = tmp["batch_sizes"].tolist()

    plt.clf()
    print("Epoch times")
    for avg_epoch_times, label in zip(epoch_times_list, labels):
        print(label)
        print(batch_sizes)
        print(avg_epoch_times)
        plt.plot(batch_sizes, np.array(avg_epoch_times) / 60, label=label)
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    print()
    print("Running times")
    plt.clf()
    for avg_run_times, label in zip(run_times_list, labels):
        print(label)
        print(batch_sizes)
        print(avg_run_times)
        plt.plot(batch_sizes, np.array(avg_run_times) / 60, label=label)
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

    plt.clf()

placement = f"../experiments/{FLAGS.speed_test_name}.npz"

if FLAGS.run_speed_test:
    run_speed_test(placement)
plot(placement)