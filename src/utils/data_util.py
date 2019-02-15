import corenlp
import os
import numpy as np
from utils.flags import FLAGS
import utils.helper as helper
import utils.tree_util as tree_util
from utils.word_embeddings_util import WordEmbeddingsUtil

class Data:
    def __init__(self):
        helper._print("================ Loading tree data ================")
        self.train_trees = tree_util.parse_trees()
        self.test_trees = tree_util.parse_trees("test")
        self.val_trees = tree_util.parse_trees("val")
        self.word_embed_util = WordEmbeddingsUtil()


class DataUtil:
    def __init__(self):
        pass
        # self.nlp_client = corenlp.CoreNLPClient(
        #     endpoint="http://localhost:8000",
        #     annotators="parse".split())

        # self.data = open(FLAGS.data_dir + 'test/test_data.txt').readlines()
        # print(self.data)
        # self.generate_input_arrays()

    def parse_sentence(self, sentence):
        return self.nlp_client.annotate(sentence)

    def generate_input_arrays(self):
        for d in self.data:
            parseTree = self.parse_sentence(d).sentence[0].parseTree

            # print(parseTree)

    def get_data(self):
        data = Data()
        return data