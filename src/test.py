import tensorflow as tf

sentence_embedding_size = 3
word_embedding_size = 2

xavier_initializer = tf.contrib.layers.xavier_initializer()

roots_pad = tf.constant([0, 1, 2])
roots = tf.constant([1, 2, 2])
roots_padded = tf.stack([roots_pad, roots], axis=1)
labels = tf.constant([[[11, 11], [12, 12], [13, 13]],
                      [[21, 21], [22, 22], [23, 23]],
                      [[31, 31], [32, 32], [33, 33]]])

word = tf.transpose(tf.constant([[0., 0.], [1., 2.], [3., 4.]]))

left_child = tf.constant([[0, 0, 1], [0, 1, 2], [0, 0, 2]])

rep_l = tf.constant([[0., 0., 0.],
                     [1., 1., 1.],
                     [11., 12., 13.]])
rep_r = tf.constant([[0., 0., 0.],
                     [2., 2., 2.],
                     [21., 22., 23.]])
rep_3 = tf.constant([[0., 0., 0.],
                     [3., 3., 3.],
                     [31., 32., 33.]])

rep_array = tf.TensorArray(
    tf.float32,
    size=0,
    dynamic_size=True,
    clear_after_read=False,
    infer_shape=False)

rep_array = rep_array.write(0, rep_l)
rep_array = rep_array.write(1, rep_r)
rep_array = rep_array.write(2, rep_3)

W = tf.get_variable(name='W', shape=[sentence_embedding_size, word_embedding_size],
                    initializer=xavier_initializer)
U_L = tf.get_variable(name='U_L', shape=[sentence_embedding_size, sentence_embedding_size],
                      initializer=xavier_initializer)
U_R = tf.get_variable(name='U_R', shape=[sentence_embedding_size, sentence_embedding_size],
                      initializer=xavier_initializer)

W_times_word = tf.matmul(W, tf.gather(word, [1], axis=1))
total = tf.matmul(W, word) + tf.matmul(U_L, rep_l) + tf.matmul(U_R, rep_r)

weights = tf.concat([W, U_L, U_R], axis=1)
inputs = tf.concat(
    [tf.gather(word, [1, 2], axis=1), tf.gather(rep_l, [1, 2], axis=0), tf.gather(rep_r, [1, 2], axis=0)], axis=0)
total_2 = tf.matmul(weights, inputs)

inputs = tf.concat([tf.gather(word, [1, 2], axis=1), tf.gather_nd(rep_array.gather([0, 0]), [[0, 2], [1, 2]]),
                    tf.gather(rep_array.gather([1, 1]), [1, 2], axis=1)], axis=1)
total_3 = tf.matmul(weights, inputs)

left_children = tf.squeeze(tf.gather(left_child, 2))


def indices():
    return [[[0, 0, 2], [0, 1, 2], [0, 2, 2]], [[1, 0, 2], [1, 1, 2], [1, 2, 2]]]


batch_size = 3
batch_indices = [[i, i] for i in range(batch_size)]


def gather_rep(step, children_indices):
    children = tf.squeeze(tf.gather(children_indices, step))
    return tf.gather_nd(rep_array.gather(children), batch_indices)


left_children = gather_rep(0, left_child)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(roots_padded))
print(sess.run(tf.gather(labels, [[0,0] [1, 1]])))
print(sess.run(tf.gather_nd(labels,roots_padded)))
print(sess.run(left_children))
print(sess.run(rep_array.gather(left_children)))
print(sess.run(tf.matmul(left_children, U_L)))
print(sess.run(tf.gather_nd(rep_array.gather(left_children), [[0, 0], [1, 1], [2, 2]])))
print(sess.run(tf.gather_nd(rep_array.gather(left_children), [[0, 2], [1, 2]])))
print(sess.run(tf.gather(word, [1], axis=1)))
print(sess.run(W))
print(sess.run(U_L))
print(sess.run(U_R))
print(sess.run(W_times_word))
print(sess.run(total))
print(sess.run(tf.shape(weights)))
print(sess.run(weights))
print(sess.run(tf.shape(inputs)))
print(sess.run(inputs))
print(sess.run(total_2))
print(sess.run(total_3))
print(sess.run(tf.gather(word, [1, 2], axis=1)))
print(sess.run(rep_array.gather([0, 1])))
print(sess.run(tf.gather_nd(rep_array.gather([0, 1]), [[0, 2], [1, 2]])))
print(sess.run(tf.gather(rep_l, [1, 2], axis=0)))
print(sess.run(rep_array.gather([0, 1])))
print(sess.run(tf.gather_nd(rep_array.gather([0, 1]), indices())))
print(sess.run(tf.matmul(gather_rep(), U_L)))
print(sess.run(tf.matmul(rep_l, U_L)))
print(sess.run(rep_l))
print(sess.run(tf.transpose(tf.concat(
    [tf.gather(word, [1, 2], axis=1), tf.transpose(tf.gather_nd(rep_array.gather([0, 0]), [[0, 0], [1, 2]])),
     tf.transpose(tf.gather_nd(rep_array.gather([1, 1]), [[0, 0], [1, 2]]))], axis=0))))
