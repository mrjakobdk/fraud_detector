

# --------------------------- Word embeddings ---------------------------

PRETRAINED_MODE = 'pretrained'
FINETUNED_MODE = 'finetuned'
TRAINED_MODE = 'trained'

GLOVE = 'glove'
GLOVE_URL = 'http://nlp.stanford.edu/data/wordvecs/'

WORD2VEC = 'word2vec'


# ----------------------------- Tree Models -----------------------------

TREE_RNN ='treeRNN'
BATCH_TREE_RNN = 'treeRNN_batch'
NEERBEK_TREE_RNN = 'treeRNN_neerbek'
TRACKER_TREE_RNN = 'treeRNN_tracker'
DEEP_RNN = 'deepRNN'
TREE_LSTM = 'treeLSTM'

ADAM_OPTIMIZER = 'adam'
ADAGRAD_OPTIMIZER = 'adagrad'

TREE_LABELS = {'ppay': '4', 'edence': '3', 'fas': '2', 'fcast': '1'}


# ------------------------------ training -------------------------------
pre_train_max_epoch = 200
pre_train_max_acc = 0.75
