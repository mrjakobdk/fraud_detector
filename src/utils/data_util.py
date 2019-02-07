import corenlp
import os
import numpy as np
from utils.flags import FLAGS

class DataUtil:
    def __init__(self):
        print(FLAGS.test)
        self.nlp_client = corenlp.CoreNLPClient(
            endpoint="http://localhost:8000",
            annotators="parse".split())

        self.data = open(FLAGS.data_dir + 'test/test_data.txt').readlines()
        print(self.data)
        self.generate_input_arrays()

    def parse_sentence(self, sentence):
        return self.nlp_client.annotate(sentence)

    def generate_input_arrays(self):
        for d in self.data:
            parseTree = self.parse_sentence(d).sentence[0].parseTree

            # print(parseTree)
