# --------------------------- Directories ---------------------------
from utils.flags import FLAGS

DATA_DIR = FLAGS.root + 'data/'
LOGS_DIR = FLAGS.root + 'logs/'
HISTORIES_DIR = FLAGS.root + 'histories/'
TRAINED_MODELS_DIR = FLAGS.root + 'trained_models/'
ENRON_DIR = DATA_DIR + 'enron/'
GLOVE_DIR = DATA_DIR + 'glove/'
WORD2VEC_DIR = DATA_DIR + 'word2vec/'
TREES_DIR = DATA_DIR + 'trees/'

# --------------------------- File Paths ---------------------------

ENRON_EMAILS_CSV_PATH = ENRON_DIR + 'emails.csv'
ENRON_EMAILS_TXT_PATH = ENRON_DIR + 'emails.txt'
ENRON_EMAILS_ZIP_PATH = ENRON_DIR + 'enron-email-dataset.zip'
ENRON_EMAILS_VOCAB_PATH = ENRON_DIR + 'vocab'
ENRON_EMAILS_COOCCUR_PATH = ENRON_DIR + 'cooccur.npy'
TREE_ALL_SENTENCES_TXT_PATH = TREES_DIR + 'all_sentences.txt'
TREE_ALL_SENTENCES_VOCAB_PATH = TREES_DIR + 'all_vocab'
TREE_SENTENCES_TXT_PATH = TREES_DIR + 'train_sentences.txt'
TREE_SENTENCES_VOCAB_PATH = TREES_DIR + 'vocab'
TREE_SENTENCES_COOCCUR_PATH = TREES_DIR + 'cooccur.npy'
GLOVE_EMBEDDING_FILE_NAME = 'glove.6B'
GLOVE_EMBEDDING_FILE_PATH = GLOVE_DIR + GLOVE_EMBEDDING_FILE_NAME + '.' + str(FLAGS.word_embedding_size) + 'd.txt'
GLOVE_EMBEDDING_ZIP_PATH = GLOVE_DIR + 'glove.zip'
FINETUNED_GLOVE_EMBEDDING_FILE_PATH = GLOVE_DIR + 'finetuned_glove.300d.txt'
WORD2VEC_EMBEDDINGS_FILE_NAME = 'GoogleNews-vectors-negative300'
WORD2VEC_EMBEDDINGS_FILE_PATH = WORD2VEC_DIR + WORD2VEC_EMBEDDINGS_FILE_NAME + '.bin'