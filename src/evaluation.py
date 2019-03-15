from utils.flags import FLAGS
from models.trees.treeRNN_batch import treeRNN_batch
from models.words_embeddings.glove import GloVe
from utils import directories, helper
from utils.data import Data
from utils.performance import Performance
import tensorflow as tf

data = Data()

word_embed = GloVe(mode=FLAGS.word_embed_mode, dimensions=FLAGS.word_embedding_size)

model_placement = directories.TRAINED_MODELS_DIR + FLAGS.model_name + "model.ckpt"

model = treeRNN_batch(data, word_embed, model_placement)

with tf.Session() as sess:
    saver = tf.train.Saver()
    model.load(sess, saver)
    helper._print("Acc:", model.accuracy(data.test_trees, sess))
    p = Performance(data.test_trees, model, sess)
    p.plot_ROC()
