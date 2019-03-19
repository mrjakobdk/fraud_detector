import tensorflow as tf

from models.words_embeddings.glove import GloVe
from models.words_embeddings.word2vec import Word2Vec
from trainers import TreeTrainer
from utils import constants
from utils.data import Data
from utils.flags import FLAGS


class Experiment():
    def __init__(self, model, data, word_embed, model_name,
                 lr=FLAGS.learning_rate,
                 lr_decay=FLAGS.lr_decay,
                 lr_end=FLAGS.learning_rate_end,
                 conv_cond=FLAGS.conv_cond,
                 batch_size=FLAGS.batch_size):
        self.model = model
        self.data = data
        self.word_embed = word_embed
        self.model_name = model_name
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_end = lr_end
        self.conv_cond = conv_cond
        self.batch_size = batch_size

    def run(self):
        with tf.Graph().as_default():
            model = self.model(self.data, self.word_embed, self.model_name,
                               learning_rate=self.lr, batch_size=self.batch_size)
            # model, load=False, config=None, batch_size=FLAGS.batch_size, epochs=FLAGS.epochs, run_times=[], epoch_times=[]
            TreeTrainer.train(model, self.lr_decay, self.lr_end, self.conv_cond)


class MadScientist():
    def __init__(self):
        self.data = Data()
        self.GloVe = GloVe(mode=constants.PRETRAINED_MODE, dimensions=FLAGS.word_embedding_size)
        self.Word2Vec = Word2Vec(mode=constants.PRETRAINED_MODE, dimensions=FLAGS.word_embedding_size)
        self.GloVe_finetuned = GloVe(mode=constants.FINETUNED_MODE, dimensions=FLAGS.word_embedding_size)
        self.Word2Vec_finetuned = Word2Vec(mode=constants.FINETUNED_MODE, dimensions=FLAGS.word_embedding_size)

        self.CPU = tf.ConfigProto(device_count={'GPU': 0})
        self.GPU = None

    def run_tree_experiments(self, list_of_experiments):
        for experiment in list_of_experiments:
            experiment.run()
