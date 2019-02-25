import tensorflow as tf

# --------------------------- Directories ---------------------------

tf.app.flags.DEFINE_string("data_dir", '../data/', "Directory for data")
tf.app.flags.DEFINE_string("logs_dir", '../logs/', "Directory for logs")
tf.app.flags.DEFINE_string("histories_dir", '../histories/', "Directory for histories")
tf.app.flags.DEFINE_string("model_name", 'rnn/', "Name for model")
tf.app.flags.DEFINE_string("glove_dir", '../data/glove/', "Directory for pre-trained GloVe word embeddings")
tf.app.flags.DEFINE_string("word2vec_dir", '../data/word2vec/', "Directory for pre-trained word2vec word embeddings")
tf.app.flags.DEFINE_string("models_dir", '../models/', "Directory for models")

# --------------------------- Training Parameters ---------------------------

tf.app.flags.DEFINE_integer("word_embedding_size", 300, "Size of the word embedding")
tf.app.flags.DEFINE_integer("sentence_embedding_size", 300, "Size of the sentence embedding")
tf.app.flags.DEFINE_integer("label_size", 2, "Number of labels")

tf.app.flags.DEFINE_integer("epochs", 200, "Number of epochs during training")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate for training")
tf.app.flags.DEFINE_float("learning_rate_end", 0.00001, "End learning rate after the total number of epoches")
tf.app.flags.DEFINE_boolean("lr_decay", False, "Use learning rate decay")
tf.app.flags.DEFINE_integer("batch_size", 5, "Batch size for training")
tf.app.flags.DEFINE_float("sensitive_weight", 1., "Weight on the sensitivity")

tf.app.flags.DEFINE_string("optimizer", 'adagrad', "Network optimizer")

tf.app.flags.DEFINE_string("weight_initializer", 'identity', "Initializer for represetation weight")
tf.app.flags.DEFINE_string("bias_initializer", 'zero', "Initializer for bias weight")

tf.app.flags.DEFINE_boolean("load_model", False, "Load a old model")

# --------------------------- Data ---------------------------

tf.app.flags.DEFINE_string("enron_emails_csv_path", '../data/enron/emails.csv', "Path for enron/emails.csv data")
tf.app.flags.DEFINE_string("enron_emails_txt_path", '../data/enron/emails.txt', "Path for enron/emails.txt data only containing the emails")

# --------------------------- Word embeddings ---------------------------

tf.app.flags.DEFINE_string("word_embed_mode", '', "Flag to switch between word embeddings modes")
tf.app.flags.DEFINE_string("glove_pretrained_mode", 'glove_pretrained', "Flag to use GloVe vectors from pretrained model")
tf.app.flags.DEFINE_string("glove_finetuned_mode", 'glove_finetuned', "Flag to use GloVe vectors from finetuned Mittens model")
tf.app.flags.DEFINE_string("glove_domain_mode", 'glove_domain', "Flag to use GloVe vectors trained on the data corpus")
tf.app.flags.DEFINE_string("word2vec_pretrained_mode", 'word2vec_pretrained', "Flag to use word2vec vectors from pretrained model")
tf.app.flags.DEFINE_string("word2vec_domain_mode", 'word2vec_pretrained', "Flag to use word2vec vectors trained on the data corpus")

tf.app.flags.DEFINE_boolean("word_embed_subset", True, "Flag whether to use a subset of the word embeddings")
tf.app.flags.DEFINE_integer("word_embed_subset_size", 100000, "Flag for the size of the subset of the word embeddings to use")
tf.app.flags.DEFINE_string("glove_url", 'http://nlp.stanford.edu/data/wordvecs/', "Base url for downloading pre-trained GloVe vectors")
tf.app.flags.DEFINE_string("glove_embedding_file", 'glove.6B', "Name of specific pre-trained GloVe vectors to be downloaded")
tf.app.flags.DEFINE_integer("word_embed_dimensions", '300', "Dimensions of pre-trained word vectors")
tf.app.flags.DEFINE_string("word2vec_embedding_file", 'GoogleNews-vectors-negative300', "Name of specific pre-trained word2vec vectors file")

# --------------------------- Etc. ---------------------------

tf.app.flags.DEFINE_boolean('verbose', True, "Global flag for 'verbose'")
tf.app.flags.DEFINE_integer('print_step_interval', 1000, "Interval to print in training")
tf.app.flags.DEFINE_boolean('run_tensorboard', True, "Flag")

# --------------------------- Init FLAGS variable ---------------------------

FLAGS = tf.app.flags.FLAGS