import os
import zipfile
from utils.flags import FLAGS
import utils.helper as helper
import numpy as np
from gensim.models.keyedvectors import KeyedVectors


class WordEmbeddingsUtil:
    def __init__(self, mode=FLAGS.glove_pretrained_mode):
        """
        :param embedding:
        :param dimensions:
         """
        helper._print_header('Getting word embeddings')

        self.dimensions = FLAGS.word_embed_dimensions
        self.mode = mode

        if mode == FLAGS.glove_pretrained_mode:
            self.embedding_file = FLAGS.glove_embedding_file
            self.embeddings, self.word2idx, self.idx2word = self.glove_embeddings()
        elif mode == FLAGS.word2vec_pretrained_mode:
            self.embedding_file = FLAGS.word2vec_embedding_file
            self.embeddings, self.word2idx, self.idx2word = self.word2vec_embeddings()

    def glove_embeddings(self):
        self.word_embed_file_path = FLAGS.glove_dir + self.embedding_file + '.' + str(self.dimensions) + 'd.txt'
        self.glove_zip_path = FLAGS.glove_dir + 'glove.zip'
        if not os.path.isdir(FLAGS.glove_dir):
            os.makedirs(FLAGS.glove_dir)
        if not os.path.isfile(self.word_embed_file_path):
            helper._print(
                '================ Downloading GloVe embedding: {0} ================'.format(self.embedding_file))
            url = 'http://nlp.stanford.edu/data/wordvecs/' + self.embedding_file + '.zip'
            helper.download(url, self.glove_zip_path)
            with zipfile.ZipFile(self.glove_zip_path, 'r') as zip:
                helper._print(f'================ Extracting glove weights from {self.glove_zip_path} ================ ')
                zip.extractall(path=FLAGS.glove_dir)
        return self.generate_indexes()

    def glove_embeddings_finetuned(self):
        pass

    def word2vec_embeddings(self):
        if not os.path.isdir(FLAGS.word2vec_dir):
            os.makedirs(FLAGS.word2vec_dir)
        self.word_embed_file_path = FLAGS.word2vec_dir + self.embedding_file + '.txt'
        if not self.dimensions == 300:
            helper._print('Only support word2vec with vectors of size 300')

        if not os.path.isfile(self.word_embed_file_path):
            binary_file_path = FLAGS.word2vec_dir + self.embedding_file + '.bin'
            if not os.path.isfile(binary_file_path):
                helper._print(
                    'Binary file not there. Download from: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM')
                exit()
            else:
                helper._print('Unpacking ' + binary_file_path)
                model = KeyedVectors.load_word2vec_format(binary_file_path, binary=True)
                helper._print('Done unpacking!')
                helper._print('Creating index files!')
                vocab_keys = model.vocab.keys()
                print(vocab_keys)
                PAD_TOKEN = 0
                word2idx = {'PAD': PAD_TOKEN}
                idx2word = {'PAD': PAD_TOKEN}
                weights = [np.random.randn(self.dimensions)]
                for index, word in enumerate(vocab_keys):
                    word2idx[word] = index + 1
                    idx2word[index + 1] = word
                    weights.append(model[word])
                    if index % FLAGS.word_embed_subset_size == 0 and index != 0:
                        helper._print(f'{index} words indexed')
                        if FLAGS.word_embed_subset:
                            break

                UNKNOWN_TOKEN = len(weights)
                word2idx['UNK'] = UNKNOWN_TOKEN
                weights.append(np.random.randn(self.dimensions))
                return np.array(weights, dtype=np.float32), word2idx, idx2word

    def generate_indexes(self):
        helper._print_header('Generating indexes for embeddings')
        PAD_TOKEN = 0
        word2idx = {'PAD': PAD_TOKEN}
        idx2word = {'PAD': PAD_TOKEN}
        weights = [np.random.randn(self.dimensions)]

        with open(self.word_embed_file_path, 'r', encoding="utf8") as file:
            for index, line in enumerate(file):
                values = line.split()  # Word and weights separated by space
                word = values[0]  # Word is first symbol on each line
                word_weights = np.asarray(values[1:], dtype=np.float32)  # Remainder of line is weights for word
                word2idx[word] = index + 1  # PAD is our zeroth index so shift by one weights.append(word_weights)
                idx2word[index + 1] = word
                weights.append(word_weights)
                if index % FLAGS.word_embed_subset_size == 0 and index != 0:
                    helper._print(f'{index} words indexed')
                    if FLAGS.word_embed_subset:
                        break
            UNKNOWN_TOKEN = len(weights)
            word2idx['UNK'] = UNKNOWN_TOKEN
            weights.append(np.random.randn(self.dimensions))
        return np.array(weights, dtype=np.float32), word2idx, idx2word

    def get_idx(self, word):
        if word in self.word2idx.keys():
            return self.word2idx[word]
        else:
            return self.word2idx['UNK']
