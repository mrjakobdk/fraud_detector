import tensorflow as tf
from utils.flags import FLAGS
import utils.helper as helper
import utils.tree_util as tree_util
import numpy as np
import sys
import os
import shutil
import webbrowser
import time

# config
sentence_embedding_size = 4
word_embedding_size = 3
label_size = 2
batch_size = 3


class word_embed_util():
    embeddings = [[0, 0, 0],
                  [0.11, 0.12, 0.13],
                  [0.21, 0.22, 0.23],
                  [0.31, 0.32, 0.33]]

    def __init__(self):
        pass

    def get_idx(self, word):
        if word == "hej":
            return 1
        if word == "david":
            return 2
        if word == "jakob":
            return 3
        return 0


class Data():
    def __init__(self):
        self.train_trees = [tree_util.parse_tree("(4 (4 Hej) (4 David))"),
                            tree_util.parse_tree("(0 (0 David) (0 Hej))"),
                            tree_util.parse_tree("(0 (0 David) (0 (0 Hej) (0 Jakob)))")]
        self.test_trees = self.train_trees
        self.val_trees = self.train_trees

        self.word_embed_util = word_embed_util()


data = Data()
# constants
embeddings = tf.constant(data.word_embed_util.embeddings)
## dummi values
rep_zero = tf.constant(0., shape=[sentence_embedding_size, batch_size])
word_zero = tf.constant(0., shape=[word_embedding_size, 1])
label_zero = tf.constant(0., shape=[label_size, batch_size])
# batch_indices = [[i, i] for i in range(batch_size)]

# tree structure placeholders
root_array = tf.placeholder(tf.int32, (None), name='root_array')
is_leaf_array = tf.placeholder(tf.bool, (None, None), name='is_leaf_array')
word_index_array = tf.placeholder(tf.int32, (None, None), name='word_index_array')
left_child_array = tf.placeholder(tf.int32, (None, None), name='left_child_array')
right_child_array = tf.placeholder(tf.int32, (None, None), name='right_child_array')
label_array = tf.placeholder(tf.int32, (None, None, label_size), name='label_array')

# initializers
xavier_initializer = tf.contrib.layers.xavier_initializer()

# word variables
W = tf.get_variable(name='W',  # shape=[sentence_embedding_size, word_embedding_size],
                    initializer=tf.constant(1., shape=[sentence_embedding_size, word_embedding_size]))

# phrase weights
U_L = tf.get_variable(name='U_L', shape=[sentence_embedding_size, sentence_embedding_size],
                      initializer=xavier_initializer)
U_R = tf.get_variable(name='U_R', shape=[sentence_embedding_size, sentence_embedding_size],
                      initializer=xavier_initializer)

# bias
b = tf.get_variable(name='b', initializer=tf.constant(100., shape=[sentence_embedding_size, 1]))

# classifier weights
V = tf.get_variable(name='V', shape=[label_size, sentence_embedding_size],
                    initializer=xavier_initializer)
b_p = tf.get_variable(name='b_p', shape=[label_size, 1], initializer=xavier_initializer)

rep_array = tf.TensorArray(
    tf.float32,
    size=0,
    dynamic_size=True,
    clear_after_read=False,
    infer_shape=False)
rep_array = rep_array.write(0, rep_zero)

word_array = tf.TensorArray(
    tf.float32,
    size=0,
    dynamic_size=True,
    clear_after_read=False,
    infer_shape=False)
word_array = word_array.write(0, word_zero)

o_array = tf.TensorArray(
    tf.float32,
    size=0,
    dynamic_size=True,
    clear_after_read=False,
    infer_shape=False)
o_array = o_array.write(0, label_zero)


def build_feed_dict_batch(roots):
    print("Batch size:", len(roots))

    node_list_list = []
    node_to_index_list = []
    for root in roots:
        node_list = []
        tree_util.depth_first_traverse(root, node_list, lambda node, node_list: node_list.append(node))
        node_list_list.append(node_list)
        node_to_index = helper.reverse_dict(node_list)
        node_to_index_list.append(node_to_index)

    feed_dict = {
        root_array: [tree_util.size_of_tree(root) for root in roots],
        is_leaf_array: helper.lists_pad([
            [False] + [node.is_leaf for node in node_list]
            for node_list in node_list_list], False),
        word_index_array: helper.lists_pad([
            [0] + [data.word_embed_util.get_idx(node.value) for node in node_list]
            for node_list in node_list_list], data.word_embed_util.get_idx("ZERO")),
        left_child_array: helper.lists_pad([
            [0] + helper.add_one(
                [node_to_index[node.left_child] if node.left_child is not None else -1 for node in node_list])
            for node_list, node_to_index in zip(node_list_list, node_to_index_list)], 0),
        right_child_array: helper.lists_pad([
            [0] + helper.add_one(
                [node_to_index[node.right_child] if node.right_child is not None else -1 for node in node_list])
            for node_list, node_to_index in zip(node_list_list, node_to_index_list)], 0),
        label_array: helper.lists_pad([
            [[0, 0]] + [node.label for node in node_list]
            for node_list in node_list_list], [0, 0])
    }

    print(feed_dict[right_child_array])
    print(feed_dict[left_child_array])
    print(feed_dict[word_index_array])

    return feed_dict


def embed_word(word_index):
    return tf.transpose(tf.nn.embedding_lookup(embeddings, word_index))


# todo check transpose perm
batch_indices = [[[j, i, j] for j in range(batch_size)] for i in range(sentence_embedding_size)]


def gather_rep(step, children_indices, rep_array):
    children = tf.squeeze(tf.gather(children_indices, step, axis=1))
    return tf.gather_nd(rep_array.gather(children), batch_indices)


def build_node(i, rep_array, word_array):
    rep_l = gather_rep(i, left_child_array, rep_array)
    rep_r = gather_rep(i, right_child_array, rep_array)
    rep_word = word_array.read(i)

    left = tf.matmul(U_L, rep_l)
    right = tf.matmul(U_R, rep_r)
    word = tf.matmul(W, rep_word)

    return tf.nn.leaky_relu(word + left + right + b)


feed_dict = build_feed_dict_batch(data.train_trees)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


def tree_construction_body(rep_array, word_array, o_array, i):
    word_index = tf.gather(word_index_array, i, axis=1)
    word_emb = embed_word(word_index)
    word_array = word_array.write(i, word_emb)

    rep = build_node(i, rep_array, word_array)
    rep_array = rep_array.write(i, rep)

    o = tf.matmul(V, rep) + b_p
    o_array = o_array.write(i, o)

    i = tf.add(i, 1)
    return rep_array, word_array, o_array, i


termination_cond = lambda rep_a, word_a, o_a, i: tf.less(i, tf.gather(tf.shape(is_leaf_array), 1))

rep_array, word_array, o_array, _ = tf.while_loop(
    cond=termination_cond,
    body=tree_construction_body,
    loop_vars=(rep_array, word_array, o_array, 1),
    parallel_iterations=1
)

#todo fix loss
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.reshape(o_array.stack(), [-1, FLAGS.label_size]),
                                               labels=label_array))

roots_pad = tf.constant([i for i in range(batch_size)])
roots_padded = tf.stack([roots_pad, root_array], axis=1)

logists = tf.gather_nd(tf.transpose(o_array.stack(), perm=[2, 0, 1]), roots_padded)
labels = tf.gather_nd(label_array, roots_padded)

logists_max = tf.argmax(logists, axis=1)
labels_max = tf.argmax(labels, axis=1)

acc = tf.equal(logists_max, labels_max)

print(sess.run(tf.gather(o_array.concat(), 0), feed_dict=feed_dict))
print(sess.run(loss, feed_dict=feed_dict))
print(sess.run(acc, feed_dict=feed_dict))

print(sess.run(roots_padded, feed_dict=feed_dict))
print(sess.run(logists, feed_dict=feed_dict))
print(sess.run(labels, feed_dict=feed_dict))
print(sess.run(root_array, feed_dict=feed_dict))
print(sess.run(label_array, feed_dict=feed_dict))
print(sess.run(tf.shape(label_array), feed_dict=feed_dict))
print(sess.run(o_array.stack(), feed_dict=feed_dict))
print(sess.run(tf.shape(o_array.stack()), feed_dict=feed_dict))
print(sess.run(tf.transpose(o_array.stack(), perm=[2, 0, 1]), feed_dict=feed_dict))
print(sess.run(tf.gather(label_array, 1, axis=1), feed_dict=feed_dict))

print(sess.run(logists_max, feed_dict=feed_dict))
print(sess.run(labels_max, feed_dict=feed_dict))

# print(sess.run(rep_zero))
# print(sess.run(word_index, feed_dict=feed_dict))
# print(sess.run(word_emb, feed_dict=feed_dict))
#
# children = tf.squeeze(tf.gather(left_child_array, i, axis=1))
# print(sess.run(children, feed_dict=feed_dict))
#
# reps = tf.gather_nd(rep_array.gather(children), batch_indices)
# print(sess.run(reps, feed_dict=feed_dict))
#
# rep_l = gather_rep(i, left_child_array, rep_array)
# print(sess.run(rep_l, feed_dict=feed_dict))
#
# rep_r = gather_rep(i, right_child_array, rep_array)
# print(sess.run(rep_r, feed_dict=feed_dict))
#
# print(sess.run(rep_array.gather([0])))
#
# word_prod = tf.matmul(W, word_emb)
# left_prod = tf.matmul(U_L, rep_l)
# right_prod = tf.matmul(U_R, rep_r)
# total = word_prod + left_prod + right_prod
#
# print(sess.run(word_prod, feed_dict=feed_dict))
# print(sess.run(left_prod, feed_dict=feed_dict))
# print(sess.run(right_prod, feed_dict=feed_dict))
# print(sess.run(total, feed_dict=feed_dict))
#
# rep = build_node(i, rep_array, word_array)
# rep_array = rep_array.write(i, rep)
#
# rep = tf.nn.leaky_relu(total + b)
#
# print(sess.run(rep, feed_dict=feed_dict))
#
#
# print(sess.run(tf.shape(rep_array.read(1)),feed_dict=feed_dict))
