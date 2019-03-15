import tensorflow as tf
import utils.data_util as data_util
import trainers.TreeTrainer as trainer

from models.trees.treeLSTM import treeLSTM
from models.trees.treeRNN import treeRNN
from models.trees.treeRNN_batch import treeRNN_batch
from models.trees.deepRNN import deepRNN
from models.trees.treeRNN_neerbek import treeRNN_neerbek
from models.trees.treeRNN_tracker import treeRNN_tracker
from models.words_embeddings.glove import GloVe
from models.words_embeddings.word2vec import Word2Vec
from utils import constants, directories
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

    if FLAGS.model_name == "":
        FLAGS.model_name = FLAGS.model + \
                           "_BathcSize" + str(FLAGS.batch_size) + \
                           "_LrStart" + str(FLAGS.learning_rate) + \
                           "_LrEnd" + str(FLAGS.learning_rate_end) + \
                           "_ExpDecay" + str(FLAGS.lr_decay) + \
                           "_ConvCond" + str(FLAGS.conv_cond) + \
                           "_WordEmbed" + str(FLAGS.word_embed_model) + '-' + str(FLAGS.word_embed_mode) + \
                           "_WordEmbedDim" + str(FLAGS.word_embedding_size) + \
                           "/"


    if FLAGS.word_embed_model == constants.WORD2VEC:
        word_embeddings = Word2Vec(mode=FLAGS.word_embed_mode, dimensions=FLAGS.word_embedding_size)
    else:  # FLAGS.word_embed_model == constants.GLOVE
        word_embeddings = GloVe(mode=FLAGS.word_embed_mode, dimensions=FLAGS.word_embedding_size)

    model_placement = directories.TRAINED_MODELS_DIR + FLAGS.model_name + "model.ckpt"
    if FLAGS.model == constants.DEEP_RNN:
        model = deepRNN(data, word_embeddings, model_placement)
    elif FLAGS.model == constants.BATCH_TREE_RNN:
        model = treeRNN_batch(data, word_embeddings, model_placement)
    elif FLAGS.model == constants.NEERBEK_TREE_RNN:
        model = treeRNN_neerbek(data, word_embeddings, model_placement)
    elif FLAGS.model == constants.TREE_LSTM:
        model = treeLSTM(data, word_embeddings, model_placement)
    elif FLAGS.model == constants.TRACKER_TREE_RNN:
        model = treeRNN_tracker(data, word_embeddings, model_placement)
    else:
        model = treeRNN(data, word_embeddings, model_placement)

    trainer.train(model, load=False, config=config)


if __name__ == "__main__":
    main()
