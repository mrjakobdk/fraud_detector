import tensorflow as tf

# --------------------------- Directories --------------------

tf.app.flags.DEFINE_string("data_dir", '../data', "Directory for data")


# --------------------------- Etc. ---------------------------


# --------------------------- Init FLAGS variable ---------------------------

FLAGS = tf.app.flags.FLAGS