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


def BEST_MODEL_DIR(model_name,
                   data_set=""): return TRAINED_MODELS_DIR + model_name + '/best_model/' if data_set == "" \
    else TRAINED_MODELS_DIR + model_name + '/best_model/' + data_set + "/"

def PRE_MODEL_DIR(model_name,
                   data_set=""): return TRAINED_MODELS_DIR + model_name + '/pre_model/' if data_set == "" \
    else TRAINED_MODELS_DIR + model_name + '/pre_model/' + data_set + "/"


def TMP_MODEL_DIR(model_name): return TRAINED_MODELS_DIR + model_name + '/tmp_model/'


ENRON_DIR = DATA_DIR + 'enron/'
GLOVE_DIR = DATA_DIR + 'glove/'
WORD2VEC_DIR = DATA_DIR + 'word2vec/'

SMALL_TREES_DIR = ENRON_DIR + 'SMALL/'
ALL_LABELS_TREES_DIR = ENRON_DIR + 'ALL/'
PPAY_TREES_DIR = ENRON_DIR + 'PPAY/'
EDENCE_TREES_DIR = ENRON_DIR + 'EDENCE/'
FAS_TREES_DIR = ENRON_DIR + 'FAS/'
FCAST_TREES_DIR = ENRON_DIR + 'FCAST/'
TREES_DIRS = {'ppay': PPAY_TREES_DIR, 'edence': EDENCE_TREES_DIR, 'fas': FAS_TREES_DIR, 'fcast': FCAST_TREES_DIR,
              'all': ALL_LABELS_TREES_DIR, 'small': SMALL_TREES_DIR}


# --------------------------- File Paths ---------------------------
def CLUSTER_FILE(model_name): return TRAINED_MODELS_DIR + model_name + '/clustering.npy'

def BEST_MODEL_FILE(model_name, data_set): return BEST_MODEL_DIR(model_name, data_set) + "model.ckpt"
def PRE_MODEL_FILE(model_name, data_set): return PRE_MODEL_DIR(model_name, data_set) + "model.ckpt"


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


def CONSOLE_FILE(model_name): return TRAINED_MODELS_DIR + model_name + '/console.txt'


PPAY_TREES_ZIP_PATH = PPAY_TREES_DIR + 'PPAY.zip'
EDENCE_TREES_ZIP_PATH = EDENCE_TREES_DIR + 'EDENCE.zip'
FAS_TREES_ZIP_PATH = FAS_TREES_DIR + 'FAS.zip'
FCAST_TREES_ZIP_PATH = FCAST_TREES_DIR + 'FCAST.zip'
TREES_ZIP_PATHS = {'ppay': PPAY_TREES_ZIP_PATH, 'edence': EDENCE_TREES_ZIP_PATH, 'fas': FAS_TREES_ZIP_PATH,
                   'fcast': FCAST_TREES_ZIP_PATH}
ENRON_TRAIN_SENTENCES_TXT_PATH = ENRON_DIR + 'train_sentences.txt'
ENRON_EMAILS_VOCAB_PATH = ENRON_DIR + 'vocab'
ENRON_EMAILS_COOCCUR_PATH = ENRON_DIR + 'cooccur.npy'

GLOVE_ENRON_VOCAB_PATH = ENRON_DIR + 'vocab'
GLOVE_ENRON_COOCCUR_PATH = ENRON_DIR + 'cooccur.npy'
GLOVE_EMBEDDING_FILE_NAME = 'glove.6B'
GLOVE_EMBEDDING_FILE_PATH = GLOVE_DIR + GLOVE_EMBEDDING_FILE_NAME + '.' + str(FLAGS.word_embedding_size) + 'd.txt'
GLOVE_EMBEDDING_ZIP_PATH = GLOVE_DIR + 'glove.zip'
FINETUNED_GLOVE_EMBEDDING_FILE_PATH = GLOVE_DIR + 'finetuned_glove.300d.txt'
TRAINED_GLOVE_EMBEDDING_FILE_PATH = GLOVE_DIR + 'trained_glove.300d.50k.txt'
WORD2VEC_EMBEDDINGS_FILE_NAME = 'GoogleNews-vectors-negative300'
WORD2VEC_EMBEDDINGS_FILE_PATH = WORD2VEC_DIR + WORD2VEC_EMBEDDINGS_FILE_NAME + '.bin'

if not os.path.exists(TRAINED_MODELS_DIR):
    os.mkdir(TRAINED_MODELS_DIR)

if not os.path.exists(MODEL_DIR(FLAGS.model_name)):
    os.mkdir(MODEL_DIR(FLAGS.model_name))


