from utils.flags import FLAGS
import numpy as np
import utils.helper as helper
import os


class Node():
    def __init__(self, is_leaf, value, label, left_child, right_child):
        self.is_leaf = is_leaf
        self.value = value
        self.label = label
        self.left_child = left_child
        self.right_child = right_child

    def to_string(self):
        if self.is_leaf:
            return "(" + str(np.argmax(self.label)) + " " + self.value + ")"
        else:
            return "(" + str(
                np.argmax(self.label)) + " " + self.left_child.to_string() + " " + self.right_child.to_string() + ")"

    def to_sentence(self):
        if self.is_leaf:
            return self.value
        else:
            return self.left_child.to_sentence() + " " + self.right_child.to_sentence()


def depth_first_traverse(node, node_list, func):
    if not node.is_leaf:
        depth_first_traverse(node.left_child, node_list, func)
        depth_first_traverse(node.right_child, node_list, func)
    func(node, node_list)


def get_preceding_lstm_index(node, start, i, preceding_lstm_index):
    i_new = i
    if not node.is_leaf:
        i_new, curr_max = get_preceding_lstm_index(node.left_child, start, i, preceding_lstm_index)
        _, curr_max = get_preceding_lstm_index(node.right_child, start, curr_max, preceding_lstm_index)
    else:
        i_new = i_new + 1
        curr_max = i_new
    preceding_lstm_index.append(i_new - 1 if i_new - 1 > start else 0)
    return i_new, curr_max


def parse_node(tokens):
    open = '('
    close = ')'
    assert tokens[0] == open, "Malformed tree"
    assert tokens[-1] == close, "Malformed tree"

    is_leaf = True
    value = None
    label = [0] * FLAGS.label_size
    label[int(int(tokens[1]) / 4)] = 1
    left_child = None
    right_child = None

    if tokens[2] == open:
        split = 3  # position after open and label
        countOpen = 1
        countClose = 0

        # Find where left child and right child split
        while countOpen != countClose:
            if tokens[split] == open:
                countOpen += 1
            if tokens[split] == close:
                countClose += 1
            split += 1

        left_child = parse_node(tokens[2:split])
        right_child = parse_node(tokens[split:-1])
        is_leaf = False
    else:
        value = ''.join(tokens[2:-1]).lower()

    return Node(is_leaf, value, label, left_child, right_child)


def parse_tree(line):
    """
    :param line: string e.g. line = "(0 (0 (0 Let) (0 (0 us) (0 (0 know) (0 (0 if) (0 (0 you) (0 (0 have) (0 (0 any) (0 questions)))))))) (0 .))"
    :return:
    """

    tokens = []
    for toks in line.strip().split():
        tokens += list(toks)

    root = parse_node(tokens)
    return root


def parse_trees(data_set="train", remove=False):  # todo maybe change input param
    """
    https://github.com/erickrf/treernn/blob/master/tree.py
    :param data_set: what dataset to use
    :return: a list of trees
    """
    file = FLAGS.data_dir + 'trees/%s.txt' % data_set
    helper._print("Loading %s trees.." % data_set)
    with open(file, 'r') as fid:
        trees = [parse_tree(l) for l in fid.readlines()]
    helper._print(len(trees), "loaded!")
    sentence_length = [count_leaf(tree) for tree in trees]
    sentence_length = np.array(sentence_length)
    helper._print("Avg length:", np.average(sentence_length))
    trees = np.array(trees)
    if remove:
        helper._print("Shorten then 90 word:",
                      int(np.sum(np.array(sentence_length) <= 90) / len(sentence_length) * 100), "%")
        helper._print("Ratio of removed labels:", ratio_of_labels(trees[np.array(sentence_length) > 90]))
        trees = np.array(
            helper.sort_by(trees[np.array(sentence_length) <= 90], sentence_length[np.array(sentence_length) <= 90]))
    return trees


def ratio_of_labels(trees):
    label_count = 0
    for tree in trees:
        if tree.label == [1, 0]:
            label_count += 1
    return label_count / len(trees)


def count_leaf(node):
    if node.is_leaf:
        return 1
    else:
        return count_leaf(node.left_child) + count_leaf(node.right_child)


def size_of_tree(node):
    if node.is_leaf:
        return 1
    else:
        return size_of_tree(node.left_child) + size_of_tree(node.right_child) + 1


def trees_to_textfile(trees, path):
    if not os.path.exists(path):
        with open(path, 'w', encoding='utf-8') as text_file:
            for tree in trees:
                line = tree.to_sentence()
                text_file.write(line + '\n')
