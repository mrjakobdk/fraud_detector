import os
import zipfile
import numpy as np

from mittens import Mittens
from mittens import GloVe as mittens_glove
from tqdm import tqdm
from models.words_embeddings.wordModel import WordModel
from utils import helper, constants, directories

class GloVe(WordModel):

    def build_pretrained_embeddings(self):
        helper._print_header('Getting pretrained GloVe embeddings')
        self.glove_download_pretrained_model()
        sentences = self.get_enron_sentences()
        vocab = self.build_vocab(sentences)
        return self.generate_indexes(vocab, directories.GLOVE_EMBEDDING_FILE_PATH)

    def build_finetuned_embeddings(self):
        helper._print_header('Getting fine-tuned GloVe embeddings')
        self.glove_download_pretrained_model()
        vocab, _, _ = self.train_and_save_finetuned_embeddings()
        return self.generate_indexes(vocab, directories.FINETUNED_GLOVE_EMBEDDING_FILE_PATH)

    def build_trained_embeddings(self):
        helper._print_header('Getting trained GloVe embeddings')
        vocab, _, _ = self.train_and_save_embeddings()
        return self.generate_indexes(vocab, directories.TRAINED_GLOVE_EMBEDDING_FILE_PATH)

    ################## HELPER FUNCTIONS ##################

    def glove_download_pretrained_model(self):
        if not os.path.isdir(directories.GLOVE_DIR):
            os.makedirs(directories.GLOVE_DIR)
        if not os.path.isfile(directories.GLOVE_EMBEDDING_FILE_PATH):
            helper._print_header(
                'Downloading GloVe embedding: {0}'.format(directories.GLOVE_EMBEDDING_FILE_NAME))
            url = constants.GLOVE_URL + directories.GLOVE_EMBEDDING_FILE_NAME + '.zip'
            helper.download(url, directories.GLOVE_EMBEDDING_ZIP_PATH)
            with zipfile.ZipFile(directories.GLOVE_EMBEDDING_ZIP_PATH, 'r') as zip:
                helper._print_header(f'Extracting glove weights from {directories.GLOVE_EMBEDDING_ZIP_PATH} ')
                zip.extractall(path=directories.GLOVE_DIR)

    def glove2dict(self, glove_filename):
        helper._print_subheader('Generating dict from pretrained GloVe embeddings')
        with open(glove_filename, 'r', encoding="utf8") as file:
            embed = {}
            lines = file.readlines()
            pbar = tqdm(
                bar_format='{percentage:.0f}%|{bar}| Elapsed: {elapsed}, Remaining: {remaining} ({n_fmt}/{total_fmt}) ',
                total=len(lines))
            for index, line in enumerate(lines):
                if index % 10000 == 0 and index != 0:
                    pbar.update(10000)
                values = line.split()
                embed[values[0]] = np.asarray(values[1:], dtype=np.float32)
            pbar.update(len(lines) % 10000)
            pbar.close()
            print()
        return embed

    def build_cooccur(self, vocab, corpus, window=10):
        helper._print_subheader("Building cooccurrence matrix")
        vocab_size = len(vocab)
        cooccurrences = np.zeros((vocab_size, vocab_size), dtype=np.float64)
        pbar = tqdm(
            bar_format='{percentage:.0f}%|{bar}| Elapsed: {elapsed}, Remaining: {remaining} ({n_fmt}/{total_fmt}) ',
            total=len(corpus))
        for i, sent in enumerate(corpus):
            if (i + 1) % 10000 == 0 and i != 0:
                pbar.update(10000)
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

        pbar.update(len(corpus) % 10000)
        pbar.close()
        print()
        helper._print(f'Done building cooccurrence matrix. Shape: {np.shape(cooccurrences)}')
        return cooccurrences

    def train_and_save_finetuned_embeddings(self):
        sentences = self.get_enron_sentences()
        vocab = self.build_vocab(sentences)
        if not os.path.isfile(directories.FINETUNED_GLOVE_EMBEDDING_FILE_PATH):
            # idx2word = {i: word for word, i in word2idx.items()}
            cooccur = self.build_cooccur(vocab, sentences)
            pretrained_embeddings = self.glove2dict(directories.GLOVE_EMBEDDING_FILE_PATH)
            helper._print(
                f'{len([v for v in vocab.keys() if v in pretrained_embeddings.keys()])} words in common with the pretrained set')
            helper._print_subheader('Building model...')
            mittens_dir = directories.GLOVE_DIR + 'mittens/'
            if not os.path.isdir(mittens_dir):
                os.makedirs(mittens_dir)
            mittens_model = Mittens(
                n=self.dimensions,
                xmax=100,
                max_iter=10000,
                display_progress=10,
                learning_rate=0.05,
                alpha=0.75,
                tol=1e-4,
                log_dir=mittens_dir,
                mittens=0.1
            )
            helper._print_subheader('Training Mittens model...')
            finetuned_embeddings = mittens_model.fit(
                cooccur,
                vocab=vocab,
                initial_embedding_dict=pretrained_embeddings)
            print()
            helper._print_subheader('Done training finetuned embeddings! Merging with pre-trained embeddings...')
            resulting_embeddings = pretrained_embeddings
            for word, weights in zip(vocab.keys(), finetuned_embeddings):
                resulting_embeddings[word] = weights
            self.dict2glove(resulting_embeddings, directories.FINETUNED_GLOVE_EMBEDDING_FILE_PATH)
            return vocab, cooccur, resulting_embeddings
        return vocab, None, None

    def train_and_save_embeddings(self):
        sentences = self.get_enron_sentences()
        vocab = self.build_vocab(sentences)
        if not os.path.isfile(directories.TRAINED_GLOVE_EMBEDDING_FILE_PATH):
            sentences = self.get_enron_sentences()
            vocab = self.build_vocab(sentences)
            cooccur = self.build_cooccur(vocab, sentences)
            helper._print_subheader('Building model...')
            glove_model = mittens_glove(
                n=300,
                xmax=100,
                max_iter=20000,
                learning_rate=0.01,
                alpha=0.75,
                tol=1e-4,
                display_progress=10,
                log_dir=directories.GLOVE_DIR + 'mittens/')
            helper._print_subheader('Training GloVE model...')
            trained_embeddings = glove_model.fit(cooccur)
            resulting_embeddings = {}
            for word, weights in zip(vocab.keys(), trained_embeddings):
                resulting_embeddings[word] = weights
            self.dict2glove(resulting_embeddings, directories.TRAINED_GLOVE_EMBEDDING_FILE_PATH)
            return vocab, cooccur, resulting_embeddings
        return vocab, None, None

    def dict2glove(self, embeddings_dict, path):
        helper._print_subheader('Saving to glove format...')
        with open(path, 'w', encoding="utf8") as file:

            pbar = tqdm(
                bar_format='{percentage:.0f}%|{bar}| Elapsed: {elapsed}, Remaining: {remaining} ({n_fmt}/{total_fmt}) ',
                total=len(embeddings_dict))
            for index, (word, weights) in enumerate(embeddings_dict.items()):
                if index % 1000 == 0 and index != 0:
                    pbar.update(1000)
                embeddings_string = word
                for weight in weights:
                    embeddings_string += ' ' + str(weight)
                file.write(embeddings_string + '\n')
            pbar.update(len(embeddings_dict) % 1000)
            pbar.close()
            print()
