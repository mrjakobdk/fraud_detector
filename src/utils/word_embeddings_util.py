from collections import Counter
import nltk
import numpy as np
import os
import sys
import csv
import re

from mittens import Mittens
from utils.flags import FLAGS
import utils.helper as helper
from utils.helper import listify
import zipfile
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import Word2Vec, KeyedVectors
from gensim.utils import simple_preprocess


class WordEmbeddingsUtil:
    def __init__(self, mode=FLAGS.glove_pretrained_mode):
        """
        :param embedding:
        :param dimensions:

        Returns a weights numpy array with dim (words, dimensions),
        a word2index mapper from word to index in the weight array,
        a idx2word mapper from index to word.
        """

        self.dimensions = FLAGS.word_embed_dimensions
        self.mode = mode

        if mode == FLAGS.glove_pretrained_mode:
            self.embedding_file = FLAGS.glove_embedding_file
            self.embeddings, self.word2idx, self.idx2word = self.glove_pretrained_embeddings()
        elif mode == FLAGS.glove_finetuned_mode:
            self.embedding_file = FLAGS.glove_embedding_file
            self.embeddings, self.word2idx, self.idx2word = self.glove_finetuned_embeddings()
        #     TODO: Parameters to word2vec. Probably not more thatn 20 epochs due to overfitting. Test window size (avg length is 20.8 words pr. sentence).
        elif mode == FLAGS.word2vec_pretrained_mode:
            self.embedding_file = FLAGS.word2vec_embedding_file
            self.embeddings, self.word2idx, self.idx2word = self.word2vec_pretrained_embeddings()
        elif mode == FLAGS.word2vec_finetuned_mode:
            self.embedding_file = FLAGS.word2vec_embedding_file
            self.embeddings, self.word2idx, self.idx2word = self.word2vec_finetuned_embeddings()
        elif mode == FLAGS.word2vec_trained_mode:
            self.embeddings, self.word2idx, self.idx2word = self.word2vec_trained_embeddings()

    def glove_pretrained_embeddings(self):
        helper._print_header('Getting pretrained GloVe embeddings')
        self.glove_download_pretrained_model()
        return self.glove_generate_indexes()

    def glove_finetuned_embeddings(self):
        helper._print_header('Getting fine-tuned GloVe embeddings')
        self.glove_download_pretrained_model()
        sentences = self.get_enron_sentences()
        vocab = helper.get_or_build(FLAGS.enron_emails_vocab_path, self.build_vocab, sentences)
        # idx2word = {i: word for word, i in word2idx.items()}
        print(len(vocab))
        cooccur = helper.get_or_build(FLAGS.enron_emails_cooccur_path, self.build_cooccur, vocab, sentences, type='numpy')
        print(cooccur)
        pretrained_embeddings = self.glove2dict(self.word_embed_file_path)
        helper._print_subheader('Starting Mittens model...')
        mittens_model = Mittens(n=self.dimensions, max_iter=1000, display_progress=1, log_dir=FLAGS.glove_dir + 'mittens/')
        finetuned_embeddings = mittens_model.fit(
            cooccur,
            vocab=vocab,
            initial_embedding_dict= pretrained_embeddings)
        print(finetuned_embeddings)

        return 'test', 'test', 'test'

    def glove2dict(self, glove_filename):
        helper._print_subheader('Generating dict from pretrained GloVe embeddings')
        with open(glove_filename) as file:
            embed = {}
            for index, line in enumerate(file):
                values = line.split()
                embed[values[0]] = np.asarray(values[1:], dtype=np.float32)
        return embed

    def glove_generate_indexes(self):
        helper._print_subheader('Generating indexes for embeddings')
        ZERO_TOKEN = 0
        word2idx = {'ZERO': ZERO_TOKEN}
        idx2word = {ZERO_TOKEN: 'ZERO'}
        weights = [np.zeros(self.dimensions)]

        with open(self.word_embed_file_path, 'r', encoding="utf8") as file:
            for index, line in enumerate(file):
                values = line.split()  # Word and weights separated by space
                word = values[0]  # Word is first symbol on each line
                word_weights = np.asarray(values[1:], dtype=np.float32)  # Remainder of line is weights for word
                word2idx[word] = index + 1  # ZERO is our zeroth index so shift by one weights.append(word_weights)
                idx2word[index + 1] = word
                weights.append(word_weights)
                if index % FLAGS.word_embed_subset_size == 0 and index != 0:
                    helper._print(f'{index} words indexed')
                    if FLAGS.word_embed_subset:
                        break
            UNKNOWN_TOKEN = len(weights)
            word2idx['UNK'] = UNKNOWN_TOKEN
            np.random.seed(240993)
            weights.append(np.random.randn(self.dimensions))

            helper._print_subheader('Indexes done!')
        return np.array(weights, dtype=np.float32), word2idx, idx2word

    def glove_download_pretrained_model(self):
        self.word_embed_file_path = FLAGS.glove_dir + self.embedding_file + '.' + str(self.dimensions) + 'd.txt'
        self.glove_zip_path = FLAGS.glove_dir + 'glove.zip'
        if not os.path.isdir(FLAGS.glove_dir):
            os.makedirs(FLAGS.glove_dir)
        if not os.path.isfile(self.word_embed_file_path):
            helper._print_header(
                'Downloading GloVe embedding: {0}'.format(self.embedding_file))
            url = 'http://nlp.stanford.edu/data/wordvecs/' + self.embedding_file + '.zip'
            helper.download(url, self.glove_zip_path)
            with zipfile.ZipFile(self.glove_zip_path, 'r') as zip:
                helper._print_header(f'Extracting glove weights from {self.glove_zip_path} ')
                zip.extractall(path=FLAGS.glove_dir)

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
        return np.array(weights, dtype=np.float32), word2idx, idx2word

    def word2vec_pretrained_embeddings(self):
        helper._print_header('Getting pretrained word2vec embeddings')
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
                sys.exit()
            else:
                helper._print_subheader('Unpacking ' + binary_file_path)
                model = KeyedVectors.load_word2vec_format(binary_file_path, binary=True)
                helper._print_subheader('Done unpacking!')
                return self.word2vec_index_keyed_vector(model)

    def word2vec_finetuned_embeddings(self):
        helper._print_header('Getting fine-tuned word2vec embeddings')
        if not os.path.isdir(FLAGS.word2vec_dir):
            os.makedirs(FLAGS.word2vec_dir)
        if os.path.isfile(FLAGS.word2vec_dir + 'finetuned_word2vec.model'):
            helper._print_subheader('Loading previously fine-tuned model...')
            finetuned_model = {}
            finetuned_model.wv = KeyedVectors.load(FLAGS.word2vec_dir + 'word2vec.model')
        else:
            if not self.dimensions == 300:
                helper._print('Only support word2vec with vectors of size 300')
                sys.exit()
            binary_file_path = FLAGS.word2vec_dir + self.embedding_file + '.bin'
            if not os.path.isfile(binary_file_path):
                helper._print(
                    'Binary file not there. Download from: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM')
                sys.exit()
            helper._print_subheader('Unpacking ' + binary_file_path)
            model = KeyedVectors.load_word2vec_format(binary_file_path, binary=True)
            helper._print_subheader('Done unpacking!')
            sentences = self.get_enron_sentences()
            finetuned_model = Word2Vec(
                size=300,
                min_count=3
            )
            helper._print_subheader('Building fine-tuned model vocab...')
            finetuned_model.build_vocab(sentences)
            helper._print_subheader('Updating with pretrained model vocab...')
            finetuned_model.build_vocab([list(model.vocab.keys())], update=True)
            helper._print_subheader('Intersection with pretrained vectors...')
            finetuned_model.intersect_word2vec_format(binary_file_path, binary=True, lockf=1.0)
            model_logger = Word2VecLogger()
            finetuned_model.train(sentences, total_examples=len(sentences), epochs=FLAGS.word2vec_finetuned_mode_epochs, callbacks=[model_logger])
            helper._print_subheader('Saving model...')
            model.save(FLAGS.word2vec_dir + 'finetuned_word2vec.model')
        return self.word2vec_index_keyed_vector(finetuned_model.wv)

    def word2vec_trained_embeddings(self):
        helper._print_header('Getting word2vec trained on Enron corpus...')
        if not os.path.isdir(FLAGS.word2vec_dir):
            os.makedirs(FLAGS.word2vec_dir)
        documents = self.get_enron_sentences()
        model_logger = Word2VecLogger()
        if os.path.isfile(FLAGS.word2vec_dir + 'word2vec.model'):
            helper._print_subheader('Loading previously trained model...')
            model = KeyedVectors.load(FLAGS.word2vec_dir + 'word2vec.model')
        else:
            helper._print_subheader('Building model...')
            model = Word2Vec(
                documents,
                size=300,
                sg=1,  # Use Skip-Gram (0 for CBOW)
                hs=0,  # Use Negative sampling. (1 for Hierarchical Softmax)
                window=10,
                min_count=3,
                workers=10,
                iter=1
            )
            helper._print_subheader('Saving untrained model...')
            model.save(FLAGS.word2vec_dir + 'word2vec.model')
        model.train(documents, total_examples=len(documents), epochs=FLAGS.word2vec_trained_mode_epochs, callbacks=[model_logger])
        helper._print_subheader('Saving model...')
        model.save(FLAGS.word2vec_dir + 'trained_word2vec.model')

        return self.word2vec_index_keyed_vector(model.wv)

    @listify
    def get_enron_sentences(self):
        helper._print_subheader('Reading ' + FLAGS.enron_emails_txt_path + '...')
        if not os.path.isfile(FLAGS.enron_emails_txt_path):
            self.load_enron_txt_data()
        with open(FLAGS.enron_emails_txt_path, 'r', encoding='utf-8') as txt_file:
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
                csv.field_size_limit((2 ** 31) - 1)
                """
            else:
                csv.field_size_limit(sys.maxsize)
        except OverflowError as e:
            # skip setting the limit for now
            pass
        if not os.path.isfile(FLAGS.enron_emails_csv_path):
            data = 'wcukierski/enron-email-dataset'
            helper._print_subheader(f'Downloading enron emails from Kaggle')
            helper.download_from_kaggle(data, FLAGS.enron_dir)
            helper._print_subheader('Download finished! Unzipping...')
            with zipfile.ZipFile(FLAGS.enron_emails_zip_path, 'r') as zip:
                zip.extractall(path=FLAGS.enron_dir)
        if not os.path.isfile(FLAGS.enron_emails_txt_path):
            helper._print_subheader('Processing emails into .txt file!')
            with open(FLAGS.enron_emails_csv_path, 'r', encoding='utf-8') as emails_csv:
                with open(FLAGS.enron_emails_txt_path, 'w', encoding='utf-8') as text_file:
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


    def build_vocab(self, corpus, min_count=FLAGS.glove_min_count):
        """
        Credit to https://github.com/hans/glove.py/blob/master/glove.py

        Returns a dictionary `w -> (i, f)`, mapping word strings to pairs of
        word ID and word corpus frequency.
        """
        helper._print_subheader('Building vocabulary from corpus')
        vocab = Counter()
        for i, doc in enumerate(corpus):
            if i % 100000 == 0 and i != 0:
                helper._print(f"{i}/{len(corpus)} sentences processed")
                break
            vocab.update(doc)
        helper._print_subheader('Done building vocabulary')
        i = 0
        word2index = {}
        for word, freq in vocab.items():
            if freq >= min_count:
                word2index[word] = i
                i += 1
        return word2index

    @listify
    def build_cooccur(self, vocab, corpus, window=10):
        helper._print_subheader("Building cooccurrence matrix")

        vocab_size = len(vocab)
        idx2word = {i: word for word, i in vocab.items()}

        cooccurrences = np.zeros((vocab_size, vocab_size), dtype=np.float64)
        helper._print('Enumerating through the corpus...')
        for i, sent in enumerate(corpus):
            if i % 10000 == 0 and i != 0:
                helper._print(f"{i}/{len(corpus)} sentences processed")
                if i == 500000:
                    break
            token_ids = [vocab[word] for word in sent if word in vocab.keys()]

            for center_i, center_id in enumerate(token_ids):
                # Collect all word IDs in left window of center word
                context_ids = token_ids[max(0, center_i - window): center_i]
                contexts_len = len(context_ids)

                for left_i, left_id in enumerate(context_ids):
                    # Distance from center word
                    distance = contexts_len - left_i

                    # Weight by inverse of distance between words
                    increment = 1.0 / float(distance)

                    # Build co-occurrence matrix symmetrically (pretend we
                    # are calculating right contexts as well)
                    cooccurrences[center_id, left_id] += increment
                    cooccurrences[left_id, center_id] += increment
        return cooccurrences


    def get_idx(self, word):
        if word in self.word2idx.keys():
            return self.word2idx[word]
        else:
            return self.word2idx['UNK']

    # def get_PCA(self, model):
    #     # fit a 2d PCA model to the vectors
    #     X = model_2[model_1.wv.vocab]
    #     pca = PCA(n_components=2)
    #     result = pca.fit_transform(X)
    #     # create a scatter plot of the projection
    #     pyplot.scatter(result[:, 0], result[:, 1])
    #     words = list(model_1.wv.vocab)
    #     for i, word in enumerate(words):
    #         pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    #     pyplot.show()


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


