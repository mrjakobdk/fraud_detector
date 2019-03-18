import os
import sys

import gensim
import numpy as np
from gensim.models import KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec

from models.words_embeddings.wordModel import WordModel
from utils import helper, constants, directories
from utils.flags import FLAGS


class Word2Vec(WordModel):
    def __init__(self, mode=constants.PRETRAINED_MODE, dimensions=FLAGS.word_embedding_size):
        self.embedding_file = directories.WORD2VEC_EMBEDDINGS_FILE_NAME
        self.word_embed_file_path = directories.WORD2VEC_EMBEDDINGS_FILE_PATH
        super(Word2Vec, self).__init__(mode, dimensions)

    def build_pretrained_embeddings(self):
        helper._print_header('Getting pretrained word2vec embeddings')
        if not os.path.isdir(directories.WORD2VEC_DIR):
            os.makedirs(directories.WORD2VEC_DIR)
        if not self.dimensions == 300:
            helper._print('Only support word2vec with vectors of size 300')

        if not os.path.isfile(self.word_embed_file_path):
            helper._print(
                'Binary file not there. Download from: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM')
            sys.exit()
        else:
            helper._print_subheader('Unpacking ' + self.word_embed_file_path)
            model = KeyedVectors.load_word2vec_format(self.word_embed_file_path, binary=True)
            helper._print_subheader('Done unpacking!')
            return self.word2vec_index_keyed_vector(model)


    def build_finetuned_embeddings(self):
        helper._print_header('Getting fine-tuned word2vec embeddings')
        if not os.path.isdir(directories.WORD2VEC_DIR):
            os.makedirs(directories.WORD2VEC_DIR)
        if os.path.isfile(directories.WORD2VEC_DIR + 'finetuned_word2vec.model'):
            helper._print_subheader('Loading previously fine-tuned model...')
            finetuned_model = {}
            finetuned_model.wv = KeyedVectors.load(directories.WORD2VEC_DIR + 'word2vec.model')
        else:
            if not self.dimensions == 300:
                helper._print('Only support word2vec with vectors of size 300')
                sys.exit()
            if not os.path.isfile(self.word_embed_file_path):
                helper._print(
                    'Binary file not there. Download from: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM')
                sys.exit()
            helper._print_subheader('Unpacking ' + self.word_embed_file_path)
            model = KeyedVectors.load_word2vec_format(self.word_embed_file_path, binary=True)
            helper._print_subheader('Done unpacking!')
            sentences = self.get_enron_sentences()
            finetuned_model = gensim.models.Word2Vec(
                size=FLAGS.word_embedding_size,
                sg=1,  # Use Skip-Gram (0 for CBOW)
                hs=0,  # Use Negative sampling. (1 for Hierarchical Softmax)
                window=FLAGS.word2vec_window,
                min_count=FLAGS.word2vec_min_count,
                workers=10,
                iter=1
            )
            helper._print_subheader('Building fine-tuned model vocab...')
            finetuned_model.build_vocab(sentences)
            helper._print_subheader('Updating with pretrained model vocab...')
            finetuned_model.build_vocab([list(model.vocab.keys())], update=True)
            helper._print_subheader('Intersection with pretrained vectors...')
            finetuned_model.intersect_word2vec_format(self.word_embed_file_path, binary=True, lockf=1.0)
            model_logger = Word2VecLogger()
            finetuned_model.train(sentences, total_examples=len(sentences), epochs=FLAGS.word2vec_epochs,
                                  callbacks=[model_logger])
            helper._print_subheader('Saving model...')
            model.save(directories.WORD2VEC_DIR + 'finetuned_word2vec.model')
        return self.word2vec_index_keyed_vector(finetuned_model.wv)


    def build_trained_embeddings(self):
        helper._print_header('Getting word2vec trained on Enron corpus...')
        if not os.path.isdir(directories.WORD2VEC_DIR):
            os.makedirs(directories.WORD2VEC_DIR)
        documents = self.get_enron_sentences()
        model_logger = Word2VecLogger()
        if os.path.isfile(directories.WORD2VEC_DIR + 'word2vec.model'):
            helper._print_subheader('Loading previously trained model...')
            model = KeyedVectors.load(directories.WORD2VEC_DIR + 'word2vec.model')
        else:
            helper._print_subheader('Building model...')
            model = gensim.models.Word2Vec(
                documents,
                size=FLAGS.word_embedding_size,
                sg=1,  # Use Skip-Gram (0 for CBOW)
                hs=0,  # Use Negative sampling. (1 for Hierarchical Softmax)
                window=FLAGS.word2vec_window,
                min_count=FLAGS.word2vec_min_count,
                workers=10,
                iter=1
            )
            helper._print_subheader('Saving untrained model...')
            model.save(directories.WORD2VEC_DIR + 'word2vec.model')
        model.train(documents, total_examples=len(documents), epochs=FLAGS.word2vec_epochs, callbacks=[model_logger])
        helper._print_subheader('Saving model...')
        model.save(directories.WORD2VEC_DIR + 'trained_word2vec.model')

        return self.word2vec_index_keyed_vector(model.wv)



    ################## HELPER FUNCTIONS ##################

    def word2vec_index_keyed_vector(self, keyed_vector):
        helper._print_subheader('Creating index files!')
        vocab_keys = keyed_vector.vocab.keys()
        ZERO_TOKEN = 0
        word2idx = {'ZERO': ZERO_TOKEN}
        idx2word = {ZERO_TOKEN: 'ZERO'}
        weights = [np.zeros(self.dimensions)]
        for index, word in enumerate(vocab_keys):
            word2idx[word] = index + 1
            idx2word[index + 1] = word
            weights.append(keyed_vector[word])
            if index % FLAGS.word_embed_subset_size == 0 and index != 0:
                helper._print(f'{index} words indexed')
                if FLAGS.word_embed_subset:
                    break

        UNKNOWN_TOKEN = len(weights)
        word2idx['UNK'] = UNKNOWN_TOKEN
        np.random.seed(240993)
        weights.append(np.random.randn(self.dimensions))

        helper._print_subheader('Index files ready!')

        # self.get_TSNE_plot(weights, [key for key in word2idx.keys()])
        return np.array(weights, dtype=np.float32), word2idx, idx2word


################## HELPER CLASSES ##################


class Word2VecLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self):
     self.epoch = 1

    def on_epoch_begin(self, model):
        helper._print(f"Epoch {self.epoch} / {model.iter}")
        self.epoch += 1

    def on_train_begin(self, model):
        helper._print_subheader(f'Training started! Going through {model.iter} epochs...')

    def on_train_end(self, model):
        helper._print_subheader('Training ended!')