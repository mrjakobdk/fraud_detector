import tensorflow as tf

flags = tf.app.flags

# --------------------------- Data ---------------------------

flags.DEFINE_string("dataset", 'small',
                    "Which dataset to use. Choose between: ppay, edence, fas, fcast, all, small (default: small)")

# --------------------------- Directories ---------------------------

flags.DEFINE_string("root", '../', "path to root folder of the project.")

# --------------------------- Training Parameters ---------------------------

flags.DEFINE_string("model_name",       '',         "Name for model")
flags.DEFINE_string("model",            "treeRNN_neerbek",  "Selecting the model to be used")
flags.DEFINE_string("act_fun",          "relu",     "Select hidden activation function (default: relu")
flags.DEFINE_float("acc_min_delta_drop",0.001,      "Mini increase in accuracy to count as increasing")
flags.DEFINE_float("acc_min_delta_conv",0.001,      "Mini increase in accuracy to count as increasing")

flags.DEFINE_integer("sentence_embedding_size", 300, "Size of the sentence embedding")
flags.DEFINE_integer("label_size", 2, "Number of labels")
flags.DEFINE_integer("deepRNN_layers", 3, "Number of layers in DeepRNN")

flags.DEFINE_integer("backoff_rate", 0, "Max number of steps in wrong direction before going back")
flags.DEFINE_integer("conv_cond", 100, "Number of epochs without find a better model(convergence condition)")
flags.DEFINE_integer("epochs", 200, "Number of epochs during training")
flags.DEFINE_integer("lr_decay", 500, "Implicit decay rate, if 0 no exp decay")
flags.DEFINE_integer("batch_size", 4, "Batch size for training")
flags.DEFINE_integer("val_freq", 1, "Frequency for validation")
flags.DEFINE_float("learning_rate", 0.1, "Learning rate for training")
flags.DEFINE_float("learning_rate_end", 0.001, "End learning rate after the total number of epoches")
flags.DEFINE_float("sensitive_weight", 1., "Weight on the sensitivity")
flags.DEFINE_float("l2_strength", 0.0001, "Strangth for l2 reg.")

flags.DEFINE_string("optimizer", 'adagrad', "Network optimizer")
flags.DEFINE_boolean("load_model", False, "Load a old model")
flags.DEFINE_boolean("use_root_loss", False, "use root or internal root loss")
flags.DEFINE_boolean("use_selective_training",
                     True, "Use selective training (default: True)")

# ---------------------------- deepRNN ----------------------------

flags.DEFINE_integer("deepRNN_depth", 3, "Trees in the deepRNN")

# ---------------------------- Selective Training ----------------------------

flags.DEFINE_integer("num_clusters", 10, "Number of clusters to use (default: 10)")
flags.DEFINE_integer("pretrain_stop_count",  50, "Stop pretraining when number of epochs without better training acc reaches this. (default: 50)")
flags.DEFINE_float("selection_cut_off", 0.27, "The expected percent to cut-off for selective training (default: 0.27)")
flags.DEFINE_boolean("mfo", True, "Whether to use MFO analysis to rate clusters. If False use prediction accuracy (default: True)")
flags.DEFINE_string("cluster_model", 'kmeans', "Which clustering model to use. (default: kmeans)")
flags.DEFINE_string("cluster_initialization", 'k-means++', "How to initialize clusters. (k-means++’, ‘random’ or an ndarray, default: k-means++)")

# --------------------------- Word embeddings ---------------------------

flags.DEFINE_integer("word_embedding_size", 300, "Size of the word embedding")
flags.DEFINE_string("word_embed_mode", 'pretrained', "Flag to switch between word embeddings modes")
flags.DEFINE_string("word_embed_model", 'glove', "Flag to switch between word embeddings modes")

flags.DEFINE_integer("glove_min_count", 100, "How many times does a word need to be present in the corpus")
flags.DEFINE_integer("glove_window", 10, "How many times does a word need to be present in the corpus")

flags.DEFINE_integer("word2vec_min_count", 100, "How many times does a word need to be present in the corpus")
flags.DEFINE_integer("word2vec_window", 10, "How many times does a word need to be present in the corpus")
flags.DEFINE_integer("word2vec_epochs", 20, "Number of epochs to train word2vec models.")

# --------------------------- Etc. ---------------------------

flags.DEFINE_boolean('verbose', True, "Global flag for 'verbose'")
flags.DEFINE_integer('print_step_interval', 1000, "Interval to print in training")
flags.DEFINE_boolean('run_tensorboard', False, "Flag")
flags.DEFINE_boolean('use_gpu', True, "Use the gpu friendly version")
flags.DEFINE_integer('num_threads', 1, "Number of threads to be use on the CPU")

# --------------------------- Experiments ---------------------------

flags.DEFINE_boolean('run_speed_test', False, "Running speed tests")
flags.DEFINE_boolean('run_batch_exp', False, "...")
flags.DEFINE_boolean('run_lr_exp', False, "...")
flags.DEFINE_boolean('run_decay_exp', False, "...")
flags.DEFINE_boolean('run_word_exp', False, "...")
flags.DEFINE_boolean('run_model_exp', False, "...")

# --------------------------- Init FLAGS variable ---------------------------

FLAGS = flags.FLAGS
