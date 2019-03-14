import utils.tree_util as tree_util
from utils import helper
from utils.flags import FLAGS
from utils.word_embeddings_util import WordEmbeddingsUtil


class Data:
    def __init__(self):
        helper._print_header("Loading tree data")
        self.train_trees = tree_util.parse_trees()
        self.test_trees = tree_util.parse_trees("test")
        self.val_trees = tree_util.parse_trees("val")

        tree_util.trees_to_textfile(self.train_trees + self.test_trees + self.val_trees, "data/trees/all_sentences.txt")

        if FLAGS.word_embed_mode == '':
            self.word_embed_util = WordEmbeddingsUtil()
        else:
            self.word_embed_util = WordEmbeddingsUtil(mode=FLAGS.word_embed_mode)
