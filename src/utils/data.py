import os
import utils.tree_util as tree_util

from utils import helper, directories
from utils.flags import FLAGS


class Data:
    def __init__(self):
        helper._print_header(f"Loading tree data ({FLAGS.dataset})")
        self.train_trees = tree_util.parse_trees(dataset=FLAGS.dataset)
        self.test_trees = tree_util.parse_trees(dataset=FLAGS.dataset, type='test')
        self.val_trees = tree_util.parse_trees(dataset=FLAGS.dataset, type="val")

        self.make_tree_text_file()


    def make_tree_text_file(self):
        if not os.path.isfile(directories.ENRON_TRAIN_SENTENCES_TXT_PATH):
            helper._print(f'Create .txt file for sentences in {directories.ENRON_TRAIN_SENTENCES_TXT_PATH}')
            all_train_trees = tree_util.parse_trees(dataset='all', type='train')
            tree_util.trees_to_textfile(list(all_train_trees), directories.ENRON_TRAIN_SENTENCES_TXT_PATH)
