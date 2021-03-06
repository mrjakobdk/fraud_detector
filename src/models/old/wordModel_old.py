import csv
import os
import re
import sys
import zipfile
import nltk
import numpy as np

from gensim.utils import simple_preprocess
from utils import constants, helper, directories, tree_util
from utils.flags import FLAGS
from utils.helper import listify
from sklearn.manifold import TSNE
from matplotlib import pyplot


class WordModel:
    # TODO: Make bars for loading stuff!
    def __init__(self, mode=constants.PRETRAINED_MODE, dimensions=FLAGS.word_embedding_size):
        self.mode = mode
        self.dimensions = dimensions
        if mode == constants.PRETRAINED_MODE:
            self.embeddings, self.word2idx, self.idx2word = self.build_pretrained_embeddings()
        elif mode == constants.FINETUNED_MODE:
            self.embeddings, self.word2idx, self.idx2word = self.build_finetuned_embeddings()
        elif mode == constants.TRAINED_MODE:
            self.embeddings, self.word2idx, self.idx2word = self.build_trained_embeddings()

    def build_pretrained_embeddings(self):
        raise NotImplementedError("Each Model must re-implement this method.")
    def build_finetuned_embeddings(self):
        raise NotImplementedError("Each Model must re-implement this method.")
    def build_trained_embeddings(self):
        raise NotImplementedError("Each Model must re-implement this method.")

    ################## HELPER FUNCTIONS ##################

    @listify
    def get_enron_sentences(self, kaggle=True, all=True):
        if kaggle:
            path = directories.ENRON_EMAILS_TXT_PATH
            if not os.path.isfile(path):
                self.load_enron_txt_data()
        else:
            if all:
                path = directories.TREE_ALL_SENTENCES_TXT_PATH
            else:
                path = directories.TREE_SENTENCES_TXT_PATH
        helper._print_subheader('Reading ' + path + '...')
        with open(path, 'r', encoding='utf-8') as txt_file:
            for index, line in enumerate(txt_file):
                if index % 1000000 == 0 and index != 0:
                    helper._print(f'{index} sentences read')
                    break
                preproccesed_line = simple_preprocess(line)
                if preproccesed_line != []:
                    yield preproccesed_line
        helper._print(f'{index} sentences read')
        helper._print_subheader('Done reading Enron email data!')

    def load_enron_txt_data(self):
        helper._print_header("Loading Enron emails")
        try:
            if os.name == 'nt':
                """
                Using sys.maxsize throws an Overflow error on Windows 64-bit platforms since internal
                representation of 'int'/'long' on Win64 is only 32-bit wide. Ideally limit on Win64
                should not exceed ((2**31)-1) as long as internal representation uses 'int' and/or 'long'
                """
                csv.field_size_limit((2 ** 31) - 1)
            else:
                csv.field_size_limit(sys.maxsize)
        except OverflowError as e:
            # skip setting the limit for now
            pass
        if not os.path.isfile(directories.ENRON_EMAILS_CSV_PATH):
            data = 'wcukierski/enron-email-dataset'
            helper._print_subheader(f'Downloading enron emails from Kaggle')
            helper.download_from_kaggle(data, directories.ENRON_DIR)
            helper._print_subheader('Download finished! Unzipping...')
            with zipfile.ZipFile(directories.ENRON_EMAILS_ZIP_PATH, 'r') as zip:
                zip.extractall(path=directories.ENRON_DIR)
        if not os.path.isfile(directories.ENRON_EMAILS_TXT_PATH):
            helper._print_subheader('Processing emails into .txt file!')
            with open(directories.ENRON_EMAILS_CSV_PATH, 'r', encoding='utf-8') as emails_csv:
                with open(directories.ENRON_EMAILS_TXT_PATH, 'w', encoding='utf-8') as text_file:
                    email_reader = csv.reader(emails_csv, delimiter=",")
                    for index, row in enumerate(email_reader):
                        if index == 0:
                            continue
                        sentences = nltk.sent_tokenize(self.format_email_body(row))
                        for sent in sentences:
                            if len(sent.split(' ')) > 2:
                                text_file.write(sent + '\n')
                        if index % 100000 == 0 and index != 0:
                            helper._print(f'{index} emails processed')

        helper._print_subheader('Enron email data loaded!')

    def format_email_body(self, body):
        body = re.split(r'X-FileName[^\n]*', body[1])[1]
        body = body.split('---------------------- Forwarded by')[0]
        body = body.split('-----Original Message-----')[0]
        body = body.split('----- Original Message -----')[0]
        body = body.replace('\n', ' ') + '\n'
        return body.strip()

    def get_idx(self, word):
        if word in self.word2idx.keys():
            return self.word2idx[word]
        else:
            return self.word2idx['UNK']

    def get_TSNE_plot(self, embeddings, vocab, words=None):
        helper._print_subheader('Plotting embeddings')
        vocab = vocab
        embeddings = embeddings
        # fit a 2d PCA model to the vectors
        tsne = TSNE(
            perplexity=30, n_components=2, verbose=2, init='pca', n_iter=5000, method='exact')
        result = tsne.fit_transform(embeddings)
        # create a scatter plot of the projection
        if not words is None:
            result = np.array([[x,y,i] for i, (x,y) in enumerate(result) if vocab[i] in words], dtype=np.float64)
            pyplot.scatter(result[:, 0], result[:, 1])
            for r in result:
                pyplot.annotate(vocab[int(r[2])], xy=(r[0], r[1]))
        else:
            pyplot.scatter(result[:, 0], result[:, 1])
            for i, word in enumerate(vocab):
                pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
        pyplot.show()


