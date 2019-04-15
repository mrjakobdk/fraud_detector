from models.trees.deepRNN import deepRNN
from models.trees.treeLSTM import treeLSTM
from models.trees.treeRNN_neerbek import treeRNN_neerbek
from models.trees.treeRNN_tracker import treeRNN_tracker
from models.words_embeddings.word2vec import Word2Vec
from utils.flags import FLAGS
from models.trees.treeRNN_batch import treeRNN_batch
from models.words_embeddings.glove import GloVe
from utils import directories, helper, constants
from utils.data import Data
from utils.performance import Performance
import tensorflow as tf

data = Data()

if FLAGS.word_embed_model == constants.WORD2VEC:
    word_embeddings = Word2Vec(mode=FLAGS.word_embed_mode, dimensions=FLAGS.word_embedding_size)
else:  # FLAGS.word_embed_model == constants.GLOVE
    word_embeddings = GloVe(mode=FLAGS.word_embed_mode, dimensions=FLAGS.word_embedding_size)

model_placement = directories.TRAINED_MODELS_DIR + FLAGS.model_name + "model.ckpt"

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
elif FLAGS.model == constants.LSTM:
    model = LSTM(data, word_embeddings, model_name)

with tf.Session() as sess:
    saver = tf.train.Saver()
    model.load(sess, saver)
    helper._print("Acc:", model.accuracy(data.test_trees, sess))
    p = Performance(data.test_trees, model, sess)
    p.plot_ROC()
