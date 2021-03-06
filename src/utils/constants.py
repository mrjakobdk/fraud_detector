

# --------------------------- Word embeddings ---------------------------
from utils.flags import FLAGS

PRETRAINED_MODE = 'pretrained'
FINETUNED_MODE = 'finetuned'
TRAINED_MODE = 'trained'

GLOVE = 'glove'
GLOVE_URL = 'http://nlp.stanford.edu/data/'

WORD2VEC = 'word2vec'
FASTTEXT = 'fasttext'
FASTTEXT_CRAWL_URL = 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip'



# ----------------------------- Tree Models -----------------------------

TREE_RNN ='treeRNN'
BATCH_TREE_RNN = 'treeRNN_batch'
NEERBEK_TREE_RNN = 'treeRNN_neerbek'
TRACKER_TREE_RNN = 'treeRNN_tracker'
TRACKER_TREE_LSTM = 'treeLSTM_tracker'
DEEP_RNN = 'deepRNN'
TREE_LSTM = 'treeLSTM'
LSTM = 'LSTM'

ADAM_OPTIMIZER = 'adam'
ADAGRAD_OPTIMIZER = 'adagrad'

TREE_LABELS = {'ppay': '4', 'edence': '3', 'fas': '2', 'fcast': '1'}

