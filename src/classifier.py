import os

import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping

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
from utils import constants, directories, helper, tree_util  # directories is need to construct console file
from utils.flags import FLAGS


def get_labels(trees):
    return [tree.label for tree in trees]


def main():
    _data_util = data_util.DataUtil()
    data = _data_util.get_data()

    roots_size = [tree_util.size_of_tree(root) for root in data.train_trees]
    data.train_trees = helper.sort_by(data.train_trees, roots_size)

    roots_size = [tree_util.size_of_tree(root) for root in data.val_trees]
    data.val_trees = helper.sort_by(data.val_trees, roots_size)

    roots_size = [tree_util.size_of_tree(root) for root in data.test_trees]
    data.test_trees = helper.sort_by(data.test_trees, roots_size)

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
            X_train = np.array(model.get_representation(data.train_trees, sess))
            Y_train = np.array(get_labels(data.train_trees))
            X_val = np.array(model.get_representation(data.val_trees, sess))
            Y_val = np.array(get_labels(data.val_trees))
            X_test = np.array(model.get_representation(data.test_trees, sess))
            Y_test = np.array(get_labels(data.test_trees))

    classifier = tf.keras.models.Sequential()
    classifier.add(tf.keras.layers.Dense(FLAGS.classifier_layer_size, activation=tf.nn.relu,
                                         input_shape=(FLAGS.sentence_embedding_size,)))
    for i in range(FLAGS.classifier_num_layers - 1):
        classifier.add(tf.keras.layers.Dense(FLAGS.classifier_layer_size, activation='relu',
                                             kernel_regularizer=tf.keras.regularizers.l2(
                                                 0.0001) if FLAGS.classifier_l2 else None))
        if FLAGS.classifier_dropout:
            classifier.add(tf.keras.layers.Dropout(0.2))
    classifier.add(tf.keras.layers.Dense(2, activation='softmax'))
    classifier.compile(optimizer=tf.keras.optimizers.Adagrad(0.01), loss='categorical_crossentropy',
                       metrics=['accuracy'])

    classifier.summary()

    epochs = 10000

    stop_early = EarlyStopping(monitor='val_acc', patience=epochs, min_delta=0.01)
    helper._print_header('Training classifier')
    classifier.fit(X_train, Y_train, batch_size=FLAGS.classifier_batch_size, validation_data=(X_val, Y_val),
                   epochs=epochs, callbacks=[stop_early], verbose=2)
    helper._print_subheader('Evaluation (validation)')
    classifier.evaluate(X_val, Y_val)
    helper._print_subheader('Evaluation (test)')
    classifier.evaluate(X_test, Y_test)

    # g_classifier = tf.Graph()
    # with g_classifier.as_default():


main()
