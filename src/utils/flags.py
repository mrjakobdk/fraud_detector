import tensorflow as tf

# --------------------------- Directories ---------------------------

tf.app.flags.DEFINE_string("data_dir", '../data/', "Directory for data")
tf.app.flags.DEFINE_string("logs_dir", '../logs/', "Directory for logs")
tf.app.flags.DEFINE_string("glove_dir", '../data/glove/', "Directory for pre-trained GloVe word embeddings")
tf.app.flags.DEFINE_string("model_filename", '../models/rnn.ckpt', "Directory for logs")

# --------------------------- Training Parameters ---------------------------

tf.app.flags.DEFINE_integer("word_embedding_size", 300, "Size of the word embedding")
tf.app.flags.DEFINE_integer("sentence_embedding_size", 300, "Size of the sentence embedding")
tf.app.flags.DEFINE_integer("label_size", 5, "Number of labels")

tf.app.flags.DEFINE_integer("epochs", 200, "Number of epochs during training")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate for training")
tf.app.flags.DEFINE_integer("batch_size", 50, "Batch size for training")
tf.app.flags.DEFINE_float("sensitive_weight", 1., "Weight on the sensitivity")

# --------------------------- Word embeddings ---------------------------

tf.app.flags.DEFINE_string("glove_url", 'http://nlp.stanford.edu/data/wordvecs/', "Base url for downloading pre-trained GloVe vectors")
tf.app.flags.DEFINE_string("glove_embedding", 'glove.6B', "Name of specific pre-trained GloVe vectors to be downloaded")
tf.app.flags.DEFINE_string("glove_dimensions", '300d', "Dimensions of pre-trained GloVe vectors")



# --------------------------- Etc. ---------------------------

tf.app.flags.DEFINE_boolean('verbose', True, "Global flag for 'verbose'")
tf.app.flags.DEFINE_integer('print_step_interval', 1000, "Interval to print in training")
tf.app.flags.DEFINE_boolean('run_tensorboard', True, "Flag")

# --------------------------- Init FLAGS variable ---------------------------

FLAGS = tf.app.flags.FLAGS