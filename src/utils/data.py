import utils.tree_util as tree_util
from utils import helper, directories

class Data:
    def __init__(self):
        helper._print_header("Loading tree data")
        self.train_trees = tree_util.parse_trees()
        self.test_trees = tree_util.parse_trees("test")
        self.val_trees = tree_util.parse_trees("val")

        tree_util.trees_to_textfile(list(self.train_trees) + list(self.test_trees) + list(self.val_trees), directories.TREE_SENTENCES_TXT_PATH)

