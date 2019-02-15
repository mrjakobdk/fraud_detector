import os
import zipfile
from utils.flags import FLAGS
import utils.helper as helper
import numpy as np


class WordEmbeddingsUtil:
    def __init__(self, embedding=FLAGS.glove_embedding, dimensions=FLAGS.glove_dimensions):
        """
        :param embedding:   The name of the pre-trained word embeddings made by Stanford.
                            Others options include: "glove.42B.300d.zip", "glove.840B.300d.zip" and "glove.twitter.27B.zip"
                            Note that the 840B and 42B only have 300 dimension vectors.
         """

        self.embedding = embedding
        self.dimensions_str = dimensions
        self.dimension_num = int(dimensions[:-1])

        self.glove_file_path = FLAGS.glove_dir + embedding + '.' + dimensions +'.txt'
        self.glove_zip_path = FLAGS.glove_dir + 'glove.zip'
        if not os.path.isdir(FLAGS.glove_dir):
            os.makedirs(FLAGS.glove_dir)
        if not os.path.isfile(self.glove_file_path):
            helper._print('================ Downloading GloVe embedding: {0} ================'.format(embedding))
            url = 'http://nlp.stanford.edu/data/wordvecs/' + embedding + '.zip'
            helper.download(url, self.glove_zip_path)
            with zipfile.ZipFile(self.glove_zip_path, 'r') as zip:
                helper._print(f'================ Extracting glove weights from {self.glove_zip_path} ================ ')
                zip.extractall(path=FLAGS.glove_dir)

        self.embeddings, self.word2idx, self.idx2word = self.generate_indexes()

    def generate_indexes(self):
        helper._print( '================ Generating indexes for embeddings ================')
        PAD_TOKEN = 0
        word2idx = { 'PAD': PAD_TOKEN }
        idx2word = { 'PAD': PAD_TOKEN }
        weights = [np.random.randn(self.dimension_num)]

        with open(self.glove_file_path, 'r', encoding="utf8") as file:
            for index, line in enumerate(file):
                values = line.split()  # Word and weights separated by space
                word = values[0] # Word is first symbol on each line
                word_weights = np.asarray(values[1:], dtype=np.float32) # Remainder of line is weights for word
                word2idx[word] = index + 1 # PAD is our zeroth index so shift by one weights.append(word_weights)
                idx2word[index + 1] = word
                weights.append(word_weights)
                if index%100000 == 0 and index != 0:
                    helper._print(f'{index} words indexed')
                    break # TODO: Remove when done testing
            UNKNOWN_TOKEN = len(weights)
            word2idx['UNK'] = UNKNOWN_TOKEN
            weights.append(np.random.randn(self.dimension_num))
        return np.array(weights, dtype=np.float32), word2idx, idx2word

    def get_idx(self, word):
        if word in self.word2idx.keys():
            return self.word2idx[word]
        else:
            return self.word2idx['UNK']

    def train_embeddings(self, data):
        pass

