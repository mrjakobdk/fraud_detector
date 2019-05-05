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

    def load_fastText_vectors(self):
        fin = io.open(directories.FASTTEXT_EMBEDDING_FILE_PATH, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}
        pbar = tqdm(
            bar_format='Elapsed: {elapsed} | {n_fmt} done')
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = map(float, tokens[1:])
            pbar.update(1)
        pbar.close()
        return data
