import os

import tensorflow as tf

import utils.data_util as data_util
import numpy as np
import talos as ta

from tensorflow.python.keras import backend
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.metrics import Accuracy, Recall, FalseNegatives, FalsePositives, TrueNegatives, \
    TruePositives
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.activations import relu
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.optimizers import Adam, Adagrad
from tensorflow.python.keras.regularizers import l2
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

def get_data():
    if not os.path.exists(directories.CLASSIFIER_DATA_DIR):
        os.mkdir(directories.CLASSIFIER_DATA_DIR)
    if not os.path.exists(directories.CLASSIFIER_DATA(FLAGS.model_name)):
        os.mkdir(directories.CLASSIFIER_DATA(FLAGS.model_name))
    if os.path.exists(directories.CLASSIFIER_DATA(FLAGS.model_name) + 'x_train.npy'):
        x_train = np.load(directories.CLASSIFIER_DATA(FLAGS.model_name) + 'x_train.npy')
        y_train = np.load(directories.CLASSIFIER_DATA(FLAGS.model_name) + 'y_train.npy')
        x_val = np.load(directories.CLASSIFIER_DATA(FLAGS.model_name) + 'x_val.npy')
        y_val = np.load(directories.CLASSIFIER_DATA(FLAGS.model_name) + 'y_val.npy')
        x_test = np.load(directories.CLASSIFIER_DATA(FLAGS.model_name) + 'x_test.npy')
        y_test = np.load(directories.CLASSIFIER_DATA(FLAGS.model_name) + 'y_test.npy')
    else:
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

            with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
                saver = tf.train.Saver()
                model.load_best(sess, saver, "validation")
                x_train = np.array(model.get_representation(data.train_trees, sess))
                y_train = np.array(get_labels(data.train_trees))
                x_val = np.array(model.get_representation(data.val_trees, sess))
                y_val = np.array(get_labels(data.val_trees))
                x_test = np.array(model.get_representation(data.test_trees, sess))
                y_test = np.array(get_labels(data.test_trees))
            np.save(directories.CLASSIFIER_DATA(FLAGS.model_name) + 'x_train', x_train)
            np.save(directories.CLASSIFIER_DATA(FLAGS.model_name) + 'y_train', y_train)
            np.save(directories.CLASSIFIER_DATA(FLAGS.model_name) + 'x_val', x_val)
            np.save(directories.CLASSIFIER_DATA(FLAGS.model_name) + 'y_val', y_val)
            np.save(directories.CLASSIFIER_DATA(FLAGS.model_name) + 'x_test', x_test)
            np.save(directories.CLASSIFIER_DATA(FLAGS.model_name) + 'y_test', y_test)
    return {'train': (x_train, y_train), 'val': (x_val, y_val), 'test': (x_test, y_test)}


class SaveBestModelCallback(Callback):
    best_val_acc = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs['acc_val'] > self.best_val_acc:
            self.best_val_acc = logs['acc_val']

    def on_train_end(self, logs=None):
        print('Best validation accuracy: ' + self.best_val_acc)

def train_classifier():
    data = get_data()

    classifier = tf.keras.models.Sequential()
    classifier.add(tf.keras.layers.Dense(FLAGS.classifier_layer_size, activation=tf.nn.relu,
                                         input_shape=(FLAGS.sentence_embedding_size,)))
    for i in range(FLAGS.classifier_num_layers - 1):
        classifier.add(tf.keras.layers.Dense(FLAGS.classifier_layer_size, activation='relu',
                                             kernel_regularizer=tf.keras.regularizers.l2(
                                                 0.3) if FLAGS.classifier_l2 else None))
        if FLAGS.classifier_dropout:
            classifier.add(tf.keras.layers.Dropout(0.5))
    classifier.add(tf.keras.layers.Dense(2, activation='softmax'))
    classifier.compile(
        optimizer=tf.keras.optimizers.Adagrad(0.01),
        loss='categorical_crossentropy',
        metrics=[Accuracy()]
    )

    classifier.summary()

    helper._print_header('Training classifier')

    classifier.fit(
        data['train'][0],
        data['train'][1],
        batch_size=FLAGS.classifier_batch_size,
        validation_data=(data['val'][0], data['val'][1]),
        epochs=200,
        callbacks=[
            EarlyStopping(monitor='val_accuracy', patience=25, min_delta=0.01),
            SaveBestModelCallback()],
        verbose=2
    )

def cross_validation():
    data = get_data()
    x_train, y_train = data['train']
    x_val, y_val = data['val']

    helper._print_header('Searching the parameter space')
    params1 = {
        'lr': [0.1, 0.01],
        'optimizer': [Adagrad],
        'activation': [relu],
        'dropout': [0, 0.2, 0.5],
        'regularization': [0, 0.01, 0.001],
        'loss_functions': ['categorical_crossentropy'],
        'layers': [1, 3],
        'layer_size': [100, 300],
        'batch_size': [4, 64],
    }
    params2 = {
        'lr': [0.1, 0.01],
        'optimizer': [Adagrad],
        'activation': [relu],
        'dropout': [0.2, 0.5],
        'regularization': [0.01, 0.001],
        'weights': [[2, 1], [1, 2], [1, 1], [3, 1]],
        'loss_functions': [weighted_categorical_crossentropy],
        'layers': [1, 3],
        'layer_size': [100, 300],
        'batch_size': [64],
    }
    t = ta.Scan(
        model=mlp_model,
        x=x_train,
        y=y_train,
        x_val=x_val,
        y_val=y_val,
        params=params2,
        dataset_name=FLAGS.model_name,
        experiment_no='patience_10_weighted_loss',
        clear_tf_session=False,
        print_params=False
    )

def weighted_categorical_crossentropy(weights):
    # Inspired by wassname on Github.
    weights = backend.variable(weights)
    def loss(y_true, y_pred):
        y_pred /= backend.sum(y_pred, axis=1, keepdims=True)
        y_pred = backend.clip(y_pred, backend.epsilon(), 1 - backend.epsilon())
        loss = y_true * backend.log(y_pred) * weights
        loss = -backend.sum(loss, -1)
        return loss
    return loss


def mlp_model(x_train, y_train, x_val, y_val, params):
    model = Sequential()
    model.add(Dense(params['layer_size'], activation=params['activation'], input_dim=x_train.shape[1]))
    for i in range(params['layers'] - 1):
        model.add(Dense(params['layer_size'], activation=params['activation'], kernel_regularizer=l2(params['regularization'])))
        model.add(Dropout(params['dropout']))
    model.add(Dense(2, activation='softmax'))
    model.compile(
        optimizer=params['optimizer'](params['lr']),
        loss=params['loss_functions'](params['weights']),
        metrics=[
            'accuracy',
            Recall(),
            FalseNegatives(),
            FalsePositives(),
            TrueNegatives(),
            TruePositives()
        ]
    )
    history = model.fit(
        x_train,
        y_train,
        batch_size=params['batch_size'],
        validation_data=(x_val, y_val),
        epochs=100,
        callbacks=[
            EarlyStopping(monitor='val_acc', patience=10, min_delta=0.01)
        ],
        verbose=0
    )
    return history, model

if FLAGS.classifier_cross_validation:
    cross_validation()
else:
    train_classifier()

#https://www.dlology.com/blog/how-to-do-hyperparameter-search-with-baysian-optimization-for-keras-model/