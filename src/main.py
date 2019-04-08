import os

import tensorflow as tf
import utils.data_util as data_util
import trainers.TreeTrainer as trainer

from models.trees.treeLSTM import treeLSTM
from models.trees.treeLSTM_tracker import treeLSTM_tracker
from models.trees.treeRNN import treeRNN
from models.trees.treeRNN_batch import treeRNN_batch
from models.trees.deepRNN import deepRNN
from models.trees.treeRNN_neerbek import treeRNN_neerbek
from models.trees.treeRNN_tracker import treeRNN_tracker
from models.words_embeddings.glove import GloVe
from models.words_embeddings.word2vec import Word2Vec
from utils import constants, directories #directories is need to construct console file
from utils.flags import FLAGS
from experiments import SpeedTester


def main():

    _data_util = data_util.DataUtil()
    data = _data_util.get_data()

    if FLAGS.use_gpu:
        config = None
    else:
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
    model_name = FLAGS.model_name
    if FLAGS.model_name == "":
        model_name = FLAGS.model + \
                           "_BathcSize" + str(FLAGS.batch_size) + \
                           "_LrStart" + str(FLAGS.learning_rate) + \
                           "_LrEnd" + str(FLAGS.learning_rate_end) + \
                           "_ExpDecay" + str(FLAGS.lr_decay) + \
                           "_ConvCond" + str(FLAGS.conv_cond) + \
                           "_WordEmbed" + str(FLAGS.word_embed_model) + '-' + str(FLAGS.word_embed_mode) + \
                           "_WordEmbedDim" + str(FLAGS.word_embedding_size)


    if FLAGS.word_embed_model == constants.WORD2VEC:
        word_embeddings = Word2Vec(mode=FLAGS.word_embed_mode, dimensions=FLAGS.word_embedding_size)
    else:  # FLAGS.word_embed_model == constants.GLOVE
        word_embeddings = GloVe(mode=FLAGS.word_embed_mode, dimensions=FLAGS.word_embedding_size)

    if FLAGS.model == constants.DEEP_RNN:
        model = deepRNN(data, word_embeddings, model_name)
    elif FLAGS.model == constants.BATCH_TREE_RNN:
        model = treeRNN_batch(data, word_embeddings, model_name)
    elif FLAGS.model == constants.NEERBEK_TREE_RNN:
        model = treeRNN_neerbek(data, word_embeddings, model_name)
    elif FLAGS.model == constants.TREE_LSTM:
        model = treeLSTM(data, word_embeddings, model_name)
    elif FLAGS.model == constants.TRACKER_TREE_RNN:
        model = treeRNN_tracker(data, word_embeddings, model_name)
    elif FLAGS.model == constants.TRACKER_TREE_LSTM:
        model = treeLSTM_tracker(data, word_embeddings, model_name)
    else:
        model = treeRNN(data, word_embeddings, model_name)
    # TODO: Check if MODEL_DIR is made prematurely
    load = FLAGS.load_model and os.path.exists(directories.TMP_MODEL_DIR(model_name))
    if FLAGS.use_selective_training:
        trainer.selective_train(model, load=load, gpu=FLAGS.use_gpu)
    else:
        trainer.train(model, load=load, gpu=FLAGS.use_gpu)


if __name__ == "__main__":
    main()
