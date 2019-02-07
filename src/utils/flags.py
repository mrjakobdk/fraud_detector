import tensorflow as tf

# --------------------------- Directories ---------------------------

tf.app.flags.DEFINE_string("data_dir", '../data/', "Directory for data")

# --------------------------- Parameters ---------------------------

tf.app.flags.DEFINE_integer("word_embedding_size", 300, "Size of the word embedding")
tf.app.flags.DEFINE_integer("sentence_embedding_size", 300, "Size of the sentence embedding")
tf.app.flags.DEFINE_integer("label_size", 2, "number of labels")

# --------------------------- Etc. ---------------------------

tf.app.flags.DEFINE_boolean('verbose', True, "Global flag for 'verbose'")

# --------------------------- Init FLAGS variable ---------------------------

FLAGS = tf.app.flags.FLAGS