import tensorflow as tf
import utils.helper as helper
from utils.flags import FLAGS
import os


class treeModel:
    def __init__(self, data, model_placement):
        # config
        self.data = data
        self.model_placement = model_placement

        self.build_placeholders()
        self.build_constants()
        self.build_variables()
        self.build_model()
        self.build_loss()
        self.build_accuracy()
        self.build_train_op()

    def build_constants(self):
        raise NotImplementedError("Each Model must re-implement this method.")

    def build_placeholders(self):
        raise NotImplementedError("Each Model must re-implement this method.")

    def build_variables(self):
        raise NotImplementedError("Each Model must re-implement this method.")

    def build_model(self):
        raise NotImplementedError("Each Model must re-implement this method.")

    def build_loss(self):
        raise NotImplementedError("Each Model must re-implement this method.")

    def build_accuracy(self):
        raise NotImplementedError("Each Model must re-implement this method.")

    def build_train_op(self):
        raise NotImplementedError("Each Model must re-implement this method.")

    def initialize(self, sess):
        # todo construct model folder
        sess.run(tf.global_variables_initializer())

    def load(self, sess, saver):
        helper._print("Restoring model...")
        saver.restore(sess, self.model_placement)
        helper._print("Model restored!")

    def save(self, sess, saver):
        helper._print("Saving model...")
        saver.save(sess, self.model_placement)
        helper._print("Model saved!")

    def build_feed_dict(self, batch):
        raise NotImplementedError("Each Model must re-implement this method.")

    def construct_dir(self):
        if not os.path.exists(FLAGS.models_dir):
            os.mkdir(FLAGS.models_dir)
        if not os.path.exists(FLAGS.models_dir + FLAGS.model_name):
            os.mkdir(FLAGS.models_dir + FLAGS.model_name)
