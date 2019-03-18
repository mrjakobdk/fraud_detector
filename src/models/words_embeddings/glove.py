import os
import zipfile
import numpy as np

from collections import Counter
from mittens import Mittens
from models.words_embeddings.wordModel import WordModel
from utils import helper, constants, directories
from utils.flags import FLAGS


class GloVe(WordModel):

    def build_pretrained_embeddings(self):
        helper._print_header('Getting pretrained GloVe embeddings')
        self.embedding_file = directories.GLOVE_EMBEDDING_FILE_NAME
        self.word_embed_file_path = directories.GLOVE_EMBEDDING_FILE_PATH
        self.glove_download_pretrained_model()
        return self.glove_generate_indexes()

    def build_finetuned_embeddings(self):
        helper._print_header('Getting fine-tuned GloVe embeddings')
        self.glove_download_pretrained_model()
        self.train_and_save_finetuned_embeddings()
        self.embedding_file = directories.FINETUNED_GLOVE_EMBEDDING_FILE_PATH
        self.word_embed_file_path = directories.FINETUNED_GLOVE_EMBEDDING_FILE_PATH
        return self.glove_generate_indexes()


    def build_trained_embeddings(self):
        raise NotImplementedError("Trained model for GloVe has not been implemented yet. :-(")

    ################## HELPER FUNCTIONS ##################

    def glove_generate_indexes(self):

        all_sentences = self.get_enron_sentences(kaggle=False)
        all_vocab = helper.get_or_build(directories.TREE_ALL_SENTENCES_VOCAB_PATH, self.build_vocab, all_sentences, 1)

        helper._print_subheader('Generating indexes for embeddings')
        weights = [np.zeros(self.dimensions)]
        ZERO_TOKEN = 0
        word2idx = {'ZERO': ZERO_TOKEN}
        idx2word = {ZERO_TOKEN: 'ZERO'}

        with open(self.word_embed_file_path, 'r', encoding="utf8") as file:
            for index, line in enumerate(file):
                values = line.split()  # Word and weights separated by space
                word = values[0]  # Word is first symbol on each line
                if word in all_vocab.keys():
                    word_weights = np.asarray(values[1:], dtype=np.float32)  # Remainder of line is weights for word
                    i = all_vocab[word]
                    word2idx[word] = i + 1  # ZERO is our zeroth index so shift by one weights.append(word_weights)
                    idx2word[i + 1] = word
                    weights.append(word_weights)
                if index % FLAGS.word_embed_subset_size == 0 and index != 0:
                    helper._print(f'{index} words indexed')
                    if FLAGS.word_embed_subset:
                        break
            UNKNOWN_TOKEN = len(weights)
            word2idx['UNK'] = UNKNOWN_TOKEN
            np.random.seed(240993)
            weights.append(np.random.randn(self.dimensions))

            # self.get_TSNE_plot(weights, [key for key in word2idx.keys()])

            helper._print_subheader(f'Indexes done! {len(weights) - 2} word embeddings!')
        return np.array(weights, dtype=np.float32), word2idx, idx2word

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
            for index, line in enumerate(file):
                if index % 100000 == 0 and index != 0:
                    helper._print(f'{index} words indexed')
                values = line.split()
                embed[values[0]] = np.asarray(values[1:], dtype=np.float32)
        return embed

    def build_vocab(self, corpus, min_count=FLAGS.glove_min_count):
        """
        Credit to https://github.com/hans/glove.py/blob/master/glove.py

        Returns a dictionary `w -> (i, f)`, mapping word strings to pairs of
        word ID and word corpus frequency.
        """
        helper._print_subheader('Building vocabulary from corpus')
        vocab = Counter()
        for i, doc in enumerate(corpus):
            if i % FLAGS.word_embed_subset_size == 0 and i != 0:
                helper._print(f"{i}/{len(corpus)} sentences processed")
                if FLAGS.word_embed_subset:
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

    def build_cooccur(self, vocab, corpus, window=10):
        helper._print_subheader("Building cooccurrence matrix")

        vocab_size = len(vocab)
        idx2word = {i: word for word, i in vocab.items()}

        cooccurrences = np.zeros((vocab_size, vocab_size), dtype=np.float64)
        helper._print('Enumerating through the corpus...')
        for i, sent in enumerate(corpus):
            if i % FLAGS.word_embed_subset_size/10 == 0 and i != 0:
                helper._print(f"{i}/{len(corpus)} sentences processed")
                if FLAGS.word_embed_subset:
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

    def train_and_save_finetuned_embeddings(self):
        if not os.path.isfile(directories.FINETUNED_GLOVE_EMBEDDING_FILE_PATH):
            helper._print_subheader('Finetuning GloVe embeddings using the Enron dataset.')
            sentences = self.get_enron_sentences(kaggle=False, all=False)
            vocab = helper.get_or_build(directories.TREE_SENTENCES_VOCAB_PATH, self.build_vocab, sentences)
            # idx2word = {i: word for word, i in word2idx.items()}
            helper._print(f'Done building vocabulary. Length: {len(vocab)}')
            cooccur = helper.get_or_build(directories.TREE_SENTENCES_COOCCUR_PATH, self.build_cooccur, vocab, sentences,
                                          type='numpy')
            pretrained_embeddings = self.glove2dict(directories.GLOVE_EMBEDDING_FILE_PATH)
            print(f'{len([v for v in vocab.keys() if v in pretrained_embeddings.keys()])} words in common with the pretrained set')
            helper._print_subheader('Done with cooccurrence matrix. Starting Mittens model...')
            mittens_model = Mittens(n=self.dimensions, max_iter=1000, display_progress=10,
                                    log_dir=directories.GLOVE_DIR + 'mittens/')
            finetuned_embeddings = mittens_model.fit(
                cooccur,
                vocab=vocab,
                initial_embedding_dict=pretrained_embeddings)
            helper._print_subheader('Done training finetuned embeddings! Merging with pre-trained embeddings...')
            resulting_embeddings = pretrained_embeddings
            for word, weights in zip(vocab.keys(), finetuned_embeddings):
                resulting_embeddings[word] = weights
            self.dict2glove(resulting_embeddings)



    def dict2glove(self, embeddings_dict):
        helper._print_subheader('Saving to glove format...')
        with open(directories.FINETUNED_GLOVE_EMBEDDING_FILE_PATH, 'w', encoding="utf8") as file:
            for index, (word, weights) in enumerate(embeddings_dict.items()):
                if index % 100000 == 0 and index != 0:
                    helper._print(f'{index} embeddings saved!')
                embeddings_string = word
                for weight in weights:
                    embeddings_string += ' ' + str(weight)
                file.write(embeddings_string+ '\n')