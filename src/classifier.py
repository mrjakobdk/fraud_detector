import os

import tensorflow as tf
import utils.data_util as data_util
import numpy as np
import trainers.TreeTrainer as trainer
from models.sequential.LSTM import LSTM

from models.trees.treeLSTM import treeLSTM
from models.trees.treeLSTM_tracker import treeLSTM_tracker
from models.trees.treeRNN_batch import treeRNN_batch
from models.trees.deepRNN import deepRNN
from models.trees.treeRNN_neerbek import treeRNN_neerbek
from models.trees.treeRNN_tracker import treeRNN_tracker
from models.words_embeddings.fastText import FastText
from models.words_embeddings.glove import GloVe
from models.words_embeddings.word2vec import Word2Vec
from utils import constants, directories, helper  # directories is need to construct console file
from utils.flags import FLAGS


def get_labels(trees):
    return [tree.label for tree in trees]


def main():

    _data_util = data_util.DataUtil()
    data = _data_util.get_data()

    if FLAGS.use_gpu:
        config = None
    else:
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )

    if FLAGS.word_embed_model == constants.WORD2VEC:
        word_embeddings = Word2Vec(mode=FLAGS.word_embed_mode, dimensions=FLAGS.word_embedding_size)
    elif FLAGS.word_embed_model == constants.FASTTEXT:
        word_embeddings = FastText(mode=FLAGS.word_embed_mode, dimensions=FLAGS.word_embedding_size)
    else:  # FLAGS.word_embed_model == constants.GLOVE
        word_embeddings = GloVe(mode=FLAGS.word_embed_mode, dimensions=FLAGS.word_embedding_size)

    g_tree = tf.Graph()
    with g_tree.as_default():
        model = None
        if FLAGS.model == constants.DEEP_RNN:
            model = deepRNN(data, word_embeddings, FLAGS.model_name)
        elif FLAGS.model == constants.BATCH_TREE_RNN:
            model = treeRNN_batch(data, word_embeddings, FLAGS.model_name)
        elif FLAGS.model == constants.NEERBEK_TREE_RNN:
            model = treeRNN_neerbek(data, word_embeddings, FLAGS.model_name)
        elif FLAGS.model == constants.TREE_LSTM:
            model = treeLSTM(data, word_embeddings, FLAGS.model_name)
        elif FLAGS.model == constants.TRACKER_TREE_RNN:
            model = treeRNN_tracker(data, word_embeddings, FLAGS.model_name)
        elif FLAGS.model == constants.TRACKER_TREE_LSTM:
            model = treeLSTM_tracker(data, word_embeddings, FLAGS.model_name)
        elif FLAGS.model == constants.LSTM:
            model = LSTM(data, word_embeddings, FLAGS.model_name)

        with tf.Session(config=None) as sess:
            saver = tf.train.Saver()
            model.load_best(sess, saver, "validation")
            X_train = model.get_representation(data.train_trees, sess)
            Y_train = get_labels(data.train_trees)
            X_val = model.get_representation(data.val_trees, sess)
            Y_val = get_labels(data.train_trees)
            X_test = model.get_representation(data.test_trees, sess)
            Y_test = get_labels(data.train_trees)


    #g_classifier = tf.Graph()
    #with g_classifier.as_default():

main()