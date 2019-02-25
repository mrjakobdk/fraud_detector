import corenlp
import os
import sys
import numpy as np
import csv
import re
from utils.flags import FLAGS
import utils.helper as helper
import utils.tree_util as tree_util
from utils.word_embeddings_util import WordEmbeddingsUtil

class Data:
    def __init__(self):
        helper._print_header("Loading tree data")
        self.train_trees = tree_util.parse_trees()
        self.test_trees = tree_util.parse_trees("test")
        self.val_trees = tree_util.parse_trees("val")

        helper._print_header("Loading enron emails")
        # self.text_data = self.load_enron_txt_data()

        if FLAGS.word_embed_mode == '':
            self.word_embed_util = WordEmbeddingsUtil()
        else:
            self.word_embed_util = WordEmbeddingsUtil(mode=FLAGS.word_embed_mode)


    def load_enron_txt_data(self):
        csv.field_size_limit(sys.maxsize)
        if not os.path.isfile(FLAGS.enron_emails_txt_path):
            helper._print('---------------- Processing emails into .txt file! ----------------')
            with open(FLAGS.enron_emails_csv_path, 'r', encoding='utf-8') as emails_csv:
                with open(FLAGS.enron_emails_txt_path, 'w', encoding='utf-8') as text_file:
                    email_reader = csv.reader(emails_csv, delimiter=",")
                    for index, row in enumerate(email_reader):
                        if index == 0:
                            continue
                        body = re.split(r'X-FileName[^\n]*', row[1])[1]
                        text_file.write(body)

                        if index % 100000 == 0 and index != 0:
                            helper._print(f'{index} emails processed')
        else:
            return open(FLAGS.enron_emails_txt_path, 'r', encoding='utf-8')



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