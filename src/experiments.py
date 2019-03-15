from experiments import SpeedTester
from experiments.MadScientist import MadScientist, Experiment
from models.trees.deepRNN import deepRNN
from models.trees.treeLSTM import treeLSTM
from models.trees.treeRNN_neerbek import treeRNN_neerbek
from models.trees.treeRNN_tracker import treeRNN_tracker
from utils import data_util
from utils.data import Data
from utils.flags import FLAGS

if FLAGS.run_speed_test:
    #batch_sizes = [2 ** i for i in range(1, 10)]
    # SpeedTester.run1(
    #     [treeRNN, treeRNN_GPU],
    #     batch_sizes,
    #     [tf.ConfigProto(device_count={'GPU': 0}), None]
    # )
    #SpeedTester.plot1(batch_sizes, ["Neerbek - CPU", "Neerbek - GPU", "Our - CPU", "Our - GPU"])

    SpeedTester.run2()
    SpeedTester.plot2()


ms = MadScientist()

if FLAGS.run_batch_exp:
    ms.run_tree_experiments([
        Experiment(model=treeRNN_neerbek, data=ms.data, word_embed=ms.GloVe, batch_size=batch_size,
                   model_placement=FLAGS.models_dir + "run_batch_exp_" + str(batch_size) + "/model.ckpt")
        for batch_size in [1, 2, 25, 50, 100, 200]
    ])

if FLAGS.run_lr_exp:
    ms.run_tree_experiments([
        Experiment(model=treeRNN_neerbek, data=ms.data, word_embed=ms.GloVe, lr=lr,
                   model_placement=FLAGS.models_dir + "run_lr_exp_" + str(lr) + "/model.ckpt")
        for lr in [0.01, 0.001, 0.0001, 0.00001]
    ])

if FLAGS.run_decay_exp:
    ms.run_tree_experiments([
        Experiment(model=treeRNN_neerbek, data=ms.data, word_embed=ms.GloVe, lr=lr,
                   model_placement=FLAGS.models_dir + "run_decay_exp_" + str(lr) + "/model.ckpt")
        for lr in [0.01, 0.001, 0.0001, 0.00001]
    ])

if FLAGS.run_word_exp:
    ms.run_tree_experiments([
        Experiment(model=treeRNN_neerbek, data=ms.data, word_embed=word_embed,
                   model_placement=FLAGS.models_dir + "run_word_exp_" + word_embed.__class__.__name__ + "/model.ckpt")
        for word_embed in [ms.GloVe, ms.Word2Vec, ms.GloVe_finetuned, ms.Word2Vec_finetuned]
    ])

if FLAGS.run_model_exp:
    ms.run_tree_experiments([
        Experiment(model=model, data=ms.data, word_embed=ms.GloVe,
                   model_placement=FLAGS.models_dir + "run_model_exp" + model.__class__.__name__ + "/model.ckpt")
        for model in [treeRNN_neerbek, deepRNN, treeLSTM, treeRNN_tracker]
    ])
