import os
import sys
import csv
import re
from utils.flags import FLAGS
import utils.helper as helper
import utils.tree_util as tree_util
from utils.word_embeddings_util import WordEmbeddingsUtil
import zipfile

class Data:
    def __init__(self):
        helper._print_header("Loading tree data")
        self.train_trees = tree_util.parse_trees()
        self.test_trees = tree_util.parse_trees("test")
        self.val_trees = tree_util.parse_trees("val")

        helper._print_header("Loading Enron emails")
        self.text_data = self.load_enron_txt_data()
        helper._print_subheader('Enron email data loaded!')

        if FLAGS.word_embed_mode == '':
            self.word_embed_util = WordEmbeddingsUtil()
        else:
            self.word_embed_util = WordEmbeddingsUtil(mode=FLAGS.word_embed_mode)


    def load_enron_txt_data(self):
        csv.field_size_limit(sys.maxsize)
        if not os.path.isfile(FLAGS.enron_emails_csv_path):
            data = 'wcukierski/enron-email-dataset'
            helper._print_subheader(f'Downloading enron emails from Kaggle')
            helper.download_from_kaggle(data, FLAGS.enron_dir)
            helper._print_subheader('Download finished! Unzipping...')
            with zipfile.ZipFile(FLAGS.enron_emails_zip_path, 'r') as zip:
                zip.extractall(path=FLAGS.enron_dir)
        if not os.path.isfile(FLAGS.enron_emails_txt_path):
            helper._print('---------------- Processing emails into .txt file! ----------------')
            with open(FLAGS.enron_emails_csv_path, 'r', encoding='utf-8') as emails_csv:
                with open(FLAGS.enron_emails_txt_path, 'w', encoding='utf-8') as text_file:
                    email_reader = csv.reader(emails_csv, delimiter=",")
                    for index, row in enumerate(email_reader):
                        if index == 0:
                            continue
                        body = re.split(r'X-FileName[^\n]*', row[1])[1]
                        body = body.split('---------------------- Forwarded by')[0]
                        body = body.split('-----Original Message-----')[0]
                        body = body.replace('\n', ' ') + '\n'
                        text_file.write(body)

                        if index % 100000 == 0 and index != 0:
                            helper._print(f'{index} emails processed')
        return open(FLAGS.enron_emails_txt_path, 'r', encoding='utf-8')



class DataUtil:
    def __init__(self):
        pass

    def parse_sentence(self, sentence):
        return self.nlp_client.annotate(sentence)

    def generate_input_arrays(self):
        for d in self.data:
            parseTree = self.parse_sentence(d).sentence[0].parseTree

            # print(parseTree)

    def get_data(self):
        data = Data()
        return data