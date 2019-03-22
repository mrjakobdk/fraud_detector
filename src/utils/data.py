import utils.tree_util as tree_util
from utils import helper, directories
from utils.flags import FLAGS


class Data:
    def __init__(self):
        helper._print_header(f"Loading tree data ({FLAGS.dataset})")
        self.train_trees = tree_util.parse_trees()
        self.test_trees = tree_util.parse_trees("test")
        self.val_trees = tree_util.parse_trees("val")

        # TODO: Used for cooccur and vocab. Set to make from all datasets.
        tree_util.trees_to_textfile(list(self.train_trees) + list(self.test_trees) + list(self.val_trees), directories.TREE_ALL_SENTENCES_TXT_PATH)
        tree_util.trees_to_textfile(list(self.train_trees), directories.TREE_SENTENCES_TXT_PATH)

