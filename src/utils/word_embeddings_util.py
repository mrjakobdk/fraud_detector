import os
import sys
import zipfile
from utils.flags import FLAGS
import utils.helper as helper
import numpy as np
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import Word2Vec, KeyedVectors
from gensim.utils import simple_preprocess


class WordEmbeddingsUtil:
    def __init__(self, mode=FLAGS.glove_pretrained_mode):
        """
        :param embedding:
        :param dimensions:
         """

        self.dimensions = FLAGS.word_embed_dimensions
        self.mode = mode

        if mode == FLAGS.glove_pretrained_mode:
            self.embedding_file = FLAGS.glove_embedding_file
            self.embeddings, self.word2idx, self.idx2word = self.glove_pretrained_embeddings()
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
        return self.glove_generate_indexes()

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
            weights.append(np.random.randn(self.dimensions))

            helper._print_subheader('Indexes done!')
        return np.array(weights, dtype=np.float32), word2idx, idx2word

    def glove_finetuned_embeddings(self):
        helper._print_header('Getting pretrained GloVe embeddings and the fine-tuning them on the Enron corpus.')

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
            documents = [doc for doc in self.get_enron_documents()]
            finetuned_model = Word2Vec(
                size=300,
                min_count=3
            )
            helper._print_subheader('Building fine-tuned model vocab...')
            finetuned_model.build_vocab(documents)
            helper._print_subheader('Updating with pretrained model vocab...')
            finetuned_model.build_vocab([list(model.vocab.keys())], update=True)
            helper._print_subheader('Intersection with pretrained vectors...')
            finetuned_model.intersect_word2vec_format(binary_file_path, binary=True, lockf=1.0)
            model_logger = Word2VecLogger()
            finetuned_model.train(documents, total_examples=len(documents), epochs=FLAGS.word2vec_finetuned_mode_epochs, callbacks=[model_logger])
            helper._print_subheader('Saving model...')
            model.save(FLAGS.word2vec_dir + 'finetuned_word2vec.model')
        return self.word2vec_index_keyed_vector(finetuned_model.wv)


    def word2vec_trained_embeddings(self):
        helper._print_header('Getting word2vec trained on Enron corpus...')
        if not os.path.isdir(FLAGS.word2vec_dir):
            os.makedirs(FLAGS.word2vec_dir)
        documents = [doc for doc in self.get_enron_documents()]
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

    def get_enron_documents(self):
        helper._print_subheader('Reading ' + FLAGS.enron_emails_txt_path + '...')
        if os.path.isfile(FLAGS.enron_emails_txt_path):
            with open(FLAGS.enron_emails_txt_path, 'r', encoding='utf-8') as txt_file:
                for index, line in enumerate(txt_file):
                    if index % 100000 == 0 and index != 0:
                        helper._print(f'{index} files read')
                    preproccesed_line = simple_preprocess(line)
                    if preproccesed_line != []:
                        yield preproccesed_line
            helper._print(f'{index} files read')
            helper._print_subheader('Done reading Enron email data!')
        else:
            print(f'No {FLAGS.enron_emails_txt_path} file for the enron data!')
            sys.exit()



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



