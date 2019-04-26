import numpy as np

from gensim.utils import simple_preprocess
from utils import constants, helper, directories
from utils.flags import FLAGS
from utils.helper import listify
from sklearn.manifold import TSNE
from matplotlib import pyplot


class WordModel:
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
    def get_enron_sentences(self):
        """
            Generator for getting the enron data as individual sentences.
        """
        helper._print_subheader('Reading ' + directories.ENRON_TRAIN_SENTENCES_TXT_PATH + '...')
        with open(directories.ENRON_TRAIN_SENTENCES_TXT_PATH, 'r', encoding='utf-8') as txt_file:
            for index, line in enumerate(txt_file):
                if index % 1000000 == 0 and index != 0:
                    helper._print(f'{index} sentences read')
                    break
                preproccesed_line = simple_preprocess(line)
                if preproccesed_line != []:
                    yield preproccesed_line
        helper._print(f'{index} sentences read')
        helper._print_subheader('Done reading Enron email data!')

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


