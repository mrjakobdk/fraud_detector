import os

from utils.flags import FLAGS

# --------------------------- Directories ---------------------------
DATA_DIR = FLAGS.root + 'data/'
TRAINED_MODELS_DIR = FLAGS.root + 'trained_models/'
def MODEL_DIR(model_name): return TRAINED_MODELS_DIR + model_name
def LOGS_DIR(model_name): return TRAINED_MODELS_DIR + model_name + '/logs/'
def LOGS_TRAIN_DIR(model_name): return TRAINED_MODELS_DIR + model_name + '/logs/train/'
def LOGS_VAL_DIR(model_name): return TRAINED_MODELS_DIR + model_name + '/logs/val/'
def LOGS_TEST_DIR(model_name): return TRAINED_MODELS_DIR + model_name + '/logs/test/'
def HISTORIES_DIR(model_name): return TRAINED_MODELS_DIR + model_name + '/histories/'
def PLOTS_DIR(model_name): return TRAINED_MODELS_DIR + model_name + '/plots/'
def BEST_MODEL_DIR(model_name): return TRAINED_MODELS_DIR + model_name + '/best_model/'
def TMP_MODEL_DIR(model_name): return TRAINED_MODELS_DIR + model_name + '/tmp_model/'
ENRON_DIR = DATA_DIR + 'enron/'
GLOVE_DIR = DATA_DIR + 'glove/'
WORD2VEC_DIR = DATA_DIR + 'word2vec/'
TREES_DIR = DATA_DIR + 'trees/'


# --------------------------- File Paths ---------------------------
def BEST_MODEL_FILE(model_name): return BEST_MODEL_DIR(model_name) + "model.ckpt"
def TMP_MODEL_FILE(model_name): return TMP_MODEL_DIR(model_name) + "model.ckpt"
def SPEED_FILE(model_name): return TRAINED_MODELS_DIR + model_name + '/speed.csv'
def PERFORMANCE_TEST_FILE(model_name): return TRAINED_MODELS_DIR + model_name + '/performance_test.csv'
def PERFORMANCE_TRAIN_FILE(model_name): return TRAINED_MODELS_DIR + model_name + '/performance_train.csv'
def PERFORMANCE_VAL_FILE(model_name): return TRAINED_MODELS_DIR + model_name + '/performance_val.csv'
def PARAMETERS_FILE(model_name): return TRAINED_MODELS_DIR + model_name + '/parameters.csv'
def SYS_ARG_FILE(model_name): return TRAINED_MODELS_DIR + model_name + '/sys_arg.txt'
def BEST_ACC_FILE(model_name): return TRAINED_MODELS_DIR + model_name + '/best_acc.csv'
def BEST_LOSS_FILE(model_name): return TRAINED_MODELS_DIR + model_name + '/best_loss.csv'
def HISTORIES_FILE(model_name): return TRAINED_MODELS_DIR + model_name + '/histories/history.npz'
def ROC_TEST_PLOT(model_name): return TRAINED_MODELS_DIR + model_name + '/plots/roc_test.png'
def ROC_TRAIN_PLOT(model_name): return TRAINED_MODELS_DIR + model_name + '/plots/roc_train.png'
def ROC_VAL_PLOT(model_name): return TRAINED_MODELS_DIR + model_name + '/plots/roc_val.png'
def ACC_HISTORY_PLOT(model_name): return TRAINED_MODELS_DIR + model_name + '/plots/acc_history.png'
def LOSS_HISTORY_PLOT(model_name): return TRAINED_MODELS_DIR + model_name + '/plots/loss_history.png'
def CONCOLE_FILE(model_name): return TRAINED_MODELS_DIR + model_name + 'concole.txt'
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

if not os.path.exists(TRAINED_MODELS_DIR):
    os.mkdir(TRAINED_MODELS_DIR)

if not os.path.exists(MODEL_DIR(FLAGS.model_name)):
    os.mkdir(MODEL_DIR(FLAGS.model_name))