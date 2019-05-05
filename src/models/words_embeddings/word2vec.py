import multiprocessing
import os
import sys
import gensim
import numpy as np

from gensim.models import KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm

from models.words_embeddings.wordModel import WordModel
from utils import helper, constants, directories
from utils.flags import FLAGS


class Word2Vec(WordModel):
    def __init__(self, mode=constants.PRETRAINED_MODE, dimensions=FLAGS.word_embedding_size):
        super(Word2Vec, self).__init__(mode, dimensions)

    def build_pretrained_embeddings(self):
        helper._print_header('Getting pretrained word2vec embeddings')
        path = directories.WORD2VEC_EMBEDDINGS_FILE_PATH
        sentences = self.get_enron_sentences()
        if not os.path.isdir(directories.WORD2VEC_DIR):
            os.makedirs(directories.WORD2VEC_DIR)
        if not self.dimensions == 300:
            helper._print('Only support word2vec with vectors of size 300')

        if not os.path.isfile(path):
            helper._print(
                'Binary file not there. Download from: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM')
            sys.exit()
        else:
            helper._print_subheader('Unpacking ' + path)
            model = KeyedVectors.load_word2vec_format(path, binary=True)
            helper._print_subheader('Done unpacking!')
            vocab = self.build_vocab(sentences)
            return self.word2vec_index_keyed_vector(keyed_vector=model, vocab=vocab)


    def build_finetuned_embeddings(self):
        helper._print_header('Getting fine-tuned word2vec embeddings')
        path = directories.WORD2VEC_DIR + 'finetuned_word2vec.model'
        pretrained_path = directories.WORD2VEC_EMBEDDINGS_FILE_PATH
        sentences = self.get_enron_sentences()
        if not os.path.isdir(directories.WORD2VEC_DIR):
            os.makedirs(directories.WORD2VEC_DIR)
        if os.path.isfile(path):
            helper._print_subheader('Loading previously fine-tuned model...')
            finetuned_model = {}
            finetuned_model.wv = KeyedVectors.load(path)
        else:
            if not self.dimensions == 300:
                helper._print('Only support word2vec with vectors of size 300')
                sys.exit()
            if not os.path.isfile(pretrained_path):
                helper._print(
                    'Binary file not there. Download from: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM')
                sys.exit()
            helper._print_subheader('Unpacking ' + pretrained_path)
            model = KeyedVectors.load_word2vec_format(pretrained_path, binary=True)
            helper._print_subheader('Done unpacking!')
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
            finetuned_model.intersect_word2vec_format(pretrained_path, binary=True, lockf=1.0)
            model_logger = Word2VecLogger()
            finetuned_model.train(sentences, total_examples=len(sentences), epochs=FLAGS.word2vec_epochs,
                                  callbacks=[model_logger])
            helper._print_subheader('Saving model...')
            finetuned_model.save(path)
        vocab = self.build_vocab(sentences)
        return self.word2vec_index_keyed_vector(keyed_vector=finetuned_model.wv, vocab=vocab)


    def build_trained_embeddings(self):
        helper._print_header('Getting word2vec trained on Enron corpus...')
        if not os.path.isdir(directories.WORD2VEC_DIR):
            os.makedirs(directories.WORD2VEC_DIR)
        sentences = self.get_enron_sentences()
        model_logger = Word2VecLogger()
        path = directories.WORD2VEC_DIR + 'trained_word2vec.model'
        if os.path.isfile(path):
            helper._print('Loading previously trained model...')
            word2vec_model = KeyedVectors.load(path)
        else:
            helper._print_subheader('Building model...')
            word2vec_model = gensim.models.Word2Vec(
                sentences,
                size=FLAGS.word_embedding_size,
                sg=1,  # Use Skip-Gram (0 for CBOW)
                hs=0,  # Use Negative sampling. (1 for Hierarchical Softmax)
                window=FLAGS.word2vec_window,
                min_count=FLAGS.word2vec_min_count,
                workers=10,
                iter=1
            )
            pool = multiprocessing.Pool()
            word2vec_model.train(sentences, total_examples=len(sentences), epochs=FLAGS.word2vec_epochs, callbacks=[model_logger])
            # word2vec_model.train(sentences, total_examples=len(sentences), epochs=FLAGS.word2vec_epochs)
            helper._print(f'Saving model to {path}')
            word2vec_model.save(path)
        vocab = self.build_vocab(sentences)
        return self.word2vec_index_keyed_vector(keyed_vector=word2vec_model.wv, vocab=vocab)

    ################## HELPER FUNCTIONS ##################

    def word2vec_index_keyed_vector(self, keyed_vector, vocab):
        helper._print_subheader('Creating index files!')
        vocab_keys = keyed_vector.vocab.keys()
        ZERO_TOKEN = 0
        word2idx = {'ZERO': ZERO_TOKEN}
        idx2word = {ZERO_TOKEN: 'ZERO'}
        weights = [np.zeros(self.dimensions)]
        pbar = tqdm(
            bar_format='Indexing keyed_vector |{bar}| Elapsed: {elapsed} | ({n_fmt}/{total_fmt})', total=len(vocab_keys))
        for index, word in enumerate(vocab_keys):
            if word in vocab.keys():
                word2idx[word] = index + 1
                idx2word[index + 1] = word
                weights.append(keyed_vector[word])
            pbar.update(1)

        pbar.close()
        print()

        UNKNOWN_TOKEN = len(weights)
        word2idx['UNK'] = UNKNOWN_TOKEN
        np.random.seed(240993)
        weights.append(np.random.randn(self.dimensions))

        helper._print('Index files ready!')

        # self.get_TSNE_plot(weights, [key for key in word2idx.keys()])
        return np.array(weights, dtype=np.float32), word2idx, idx2word


################## HELPER CLASSES ##################


class Word2VecLogger(CallbackAny2Vec):
    '''Callback to log information about training'''
    def __init__(self):
        self.epoch = 1

    def on_epoch_begin(self, model):
        helper._print_header(f'Epoch: {self.epoch}/{model.iter}')
        self.epoch += 1
    def on_train_begin(self, model):
        helper._print_subheader(f'Training Model ({model.iter} epochs)...')
        # self.pbar = tqdm(
        #     bar_format='{percentage:.0f}%|{bar}| Epoch: {n_fmt}, Elapsed: {elapsed}, Remaining: {remaining}',
        #     total=model.iter)

    def on_train_end(self, model):
        # self.pbar.close()
        helper._print_subheader('Training ended!')