import tensorflow as tf

flags = tf.app.flags

# --------------------------- Directories ---------------------------

flags.DEFINE_string("data_dir", '../data/', "Directory for data")
flags.DEFINE_string("logs_dir", '../logs/', "Directory for logs")
flags.DEFINE_string("enron_dir", '../data/enron/', "Directory for logs")
flags.DEFINE_string("histories_dir", '../histories/', "Directory for histories")
flags.DEFINE_string("model_name", '', "Name for model")
flags.DEFINE_string("glove_dir", '../data/glove/', "Directory for pre-trained GloVe word embeddings")
flags.DEFINE_string("word2vec_dir", '../data/word2vec/', "Directory for pre-trained word2vec word embeddings")
flags.DEFINE_string("models_dir", '../trained_models/', "Directory for trained_models")

# --------------------------- Training Parameters ---------------------------

flags.DEFINE_string("model", "treeRNN", "Selecting the model to be used")

flags.DEFINE_integer("word_embedding_size", 300, "Size of the word embedding")
flags.DEFINE_integer("sentence_embedding_size", 300, "Size of the sentence embedding")
flags.DEFINE_integer("label_size", 2, "Number of labels")
flags.DEFINE_integer("deepRNN_layers", 3, "Number of layers in DeepRNN")

flags.DEFINE_integer("conv_cond", 100, "Number of epochs without find a better model(convergence condition)")
flags.DEFINE_integer("epochs", 200, "Number of epochs during training")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate for training")
flags.DEFINE_float("learning_rate_end", 0.00001, "End learning rate after the total number of epoches")
flags.DEFINE_integer("lr_decay", 500, "Implicit decay rate, if 0 no exp decay")
flags.DEFINE_integer("batch_size", 4, "Batch size for training")
flags.DEFINE_float("sensitive_weight", 1., "Weight on the sensitivity")
flags.DEFINE_float("l2_strangth", 0.0001, "Strangth for l2 reg.")

flags.DEFINE_string("optimizer", 'adagrad', "Network optimizer")

flags.DEFINE_string("weight_initializer", 'identity', "Initializer for represetation weight")
flags.DEFINE_string("bias_initializer", 'zero', "Initializer for bias weight")

flags.DEFINE_boolean("load_model", False, "Load a old model")

# ---------------------------- deepRNN ----------------------------

flags.DEFINE_integer("deepRNN_depth", 3, "Trees in the deepRNN")

# --------------------------- Data ---------------------------

flags.DEFINE_string("enron_emails_csv_path", '../data/enron/emails.csv', "Path for enron/emails.csv data")
flags.DEFINE_string("enron_emails_txt_path", '../data/enron/emails.txt',
                    "Path for enron/emails.txt data only containing the emails")
flags.DEFINE_string("enron_emails_zip_path", '../data/enron/enron-email-dataset.zip',
                    "Path for enron/enron-email-dataset.zip")
flags.DEFINE_string("enron_emails_vocab_path", '../data/enron/vocab', "Path for enron vocabulary")
flags.DEFINE_string("enron_emails_cooccur_path", '../data/enron/cooccur.npy', "Path for enron weighted cooccurrence matrix")
flags.DEFINE_string("tree_sentences_txt_path", '../data/trees/all_sentences.txt', 'Path for tree sentences .txt file.')

"data/trees/all_sentences.txt"

# --------------------------- Word embeddings ---------------------------

flags.DEFINE_string("word_embed_mode", '', "Flag to switch between word embeddings modes")
flags.DEFINE_string("glove_pretrained_mode", 'glove_pretrained', "Flag to use GloVe vectors from pretrained model")
flags.DEFINE_string("glove_finetuned_mode", 'glove_finetuned', "Flag to use GloVe vectors from finetuned Mittens model")
flags.DEFINE_string("glove_trained_mode", 'glove_trained', "Flag to use GloVe vectors trained on the data corpus")

flags.DEFINE_integer("glove_min_count", 2, "How many times does a word need to be present in the corpus")
flags.DEFINE_string("word2vec_pretrained_mode", 'word2vec_pretrained',
                    "Flag to use word2vec vectors from pretrained model")

flags.DEFINE_string("word2vec_finetuned_mode", 'word2vec_finetuned',
                    "Flag to use word2vec vectors from pretrained model, but finetuned on the Enron dataset")
flags.DEFINE_integer("word2vec_finetuned_mode_epochs", 20, "How many epoch do we want to train the word2vec embeddings")

flags.DEFINE_string("word2vec_trained_mode", 'word2vec_trained',
                    "Flag to use word2vec vectors trained on the data corpus")
flags.DEFINE_integer("word2vec_trained_mode_epochs", 20, "How many epoch do we want to train the word2vec embeddings")

flags.DEFINE_boolean("word_embed_subset", True, "Flag whether to use a subset of the word embeddings")
flags.DEFINE_integer("word_embed_subset_size", 100000, "Flag for the size of the subset of the word embeddings to use")
flags.DEFINE_string("glove_url", 'http://nlp.stanford.edu/data/wordvecs/',
                    "Base url for downloading pre-trained GloVe vectors")
flags.DEFINE_string("glove_embedding_file", 'glove.6B', "Name of specific pre-trained GloVe vectors to be downloaded")
flags.DEFINE_integer("word_embed_dimensions", '300', "Dimensions of pre-trained word vectors")
flags.DEFINE_string("word2vec_embedding_file", 'GoogleNews-vectors-negative300',
                    "Name of specific pre-trained word2vec vectors file")

# --------------------------- Etc. ---------------------------

flags.DEFINE_boolean('verbose', True, "Global flag for 'verbose'")
flags.DEFINE_integer('print_step_interval', 1000, "Interval to print in training")
flags.DEFINE_boolean('run_tensorboard', False, "Flag")
flags.DEFINE_boolean('use_gpu', False, "Use the gpu friendly version")

flags.DEFINE_boolean('run_speed_test', False, "Running speed tests")

# --------------------------- Init FLAGS variable ---------------------------

FLAGS = flags.FLAGS
