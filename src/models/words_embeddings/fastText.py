import io
import os
import zipfile
from collections import Counter

import numpy as np

from tqdm import tqdm
from models.words_embeddings.wordModel import WordModel
from utils import constants, directories, helper
from utils.flags import FLAGS


class FastText(WordModel):

    def build_pretrained_embeddings(self):
        helper._print_header('Getting pretrained fastText embeddings')
        if self.dimensions != 300:
            raise NotImplementedError('Only word vectors of size 300 are available at this point.')
        self.download_fastText_vectors()
        sentences = self.get_enron_sentences()
        vocab = self.build_vocab(sentences)
        return self.generate_indexes(vocab, directories.FASTTEXT_EMBEDDING_FILE_PATH)

    def build_finetuned_embeddings(self):
        raise NotImplementedError('No finetuned embeddings implemented for fastText')

    def build_trained_embeddings(self):
        raise NotImplementedError('No trained embeddings implemented for fastText')


    ################## HELPER FUNCTIONS ##################

    def download_fastText_vectors(self):
        if os.path.exists(directories.FASTTEXT_EMBEDDING_FILE_PATH):
            return
        else:
            helper.download(constants.FASTTEXT_CRAWL_URL, directories.FASTTEXT_EMBEDDING_ZIP_PATH)
            with zipfile.ZipFile(directories.FASTTEXT_EMBEDDING_ZIP_PATH, 'r') as zip:
                zip.extractall(path=directories.FASTTEXT_DIR)
            return
    def build_vocab(self, corpus, min_count=FLAGS.word_min_count):
        helper._print_subheader('Building vocabulary from corpus')
        vocab = Counter()
        pbar = tqdm(
            bar_format='{percentage:.0f}%|{bar}| Elapsed: {elapsed}, Remaining: {remaining} ({n_fmt}/{total_fmt}) ',
            total=len(corpus))
        for i, doc in enumerate(corpus):
            if (i + 1) % 1000 == 0 and i != 0:
                pbar.update(1000)
            vocab.update(doc)
        pbar.update(len(corpus) % 1000)
        pbar.close()
        print()
        i = 0
        word2index = {}
        for word, freq in vocab.items():
            if freq >= min_count:
                word2index[word] = i
                i += 1

        helper._print(f'Done building vocabulary. Length: {len(word2index)}')
        return word2index
