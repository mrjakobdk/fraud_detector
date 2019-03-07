import tensorflow as tf

# config
sentence_embedding_size = 4
word_embedding_size = 3
label_size = 2
batch_size = 3

rep_array = tf.TensorArray(
    tf.float32,
    size=0,
    dynamic_size=True,
    clear_after_read=False,
    infer_shape=False)

children = tf.constant([[0, 0, 0, 0], [1, 2, 3, 0], [0, 1, 0, 0]])

# phrase rep size = 4 and is the columns
rep_0 = tf.constant([[000., 001., 002.],
                     [010., 011., 012.],
                     [020., 021., 022.],
                     [030., 031., 032.]])
rep_1 = tf.constant([[100., 101., 102.],
                     [110., 111., 112.],
                     [120., 121., 122.],
                     [130., 131., 132.]])
rep_2 = tf.constant([[200., 201., 202.],
                     [210., 211., 212.],
                     [220., 221., 222.],
                     [230., 231., 232.]])
rep_array = rep_array.write(0, rep_0)
rep_array = rep_array.write(1, rep_1)
rep_array = rep_array.write(2, rep_2)

batch_indices = [[i, i] for i in range(batch_size)]

batch_indices = [[[j, i, j] for j in range(batch_size)] for i in range(sentence_embedding_size)]
def gather_rep(step, children_indices, rep_array):
    children = tf.squeeze(tf.gather(children_indices, step, axis=1))
    return tf.gather_nd(rep_array.gather(children), batch_indices)

batch_indices = [[j, j] for j in range(batch_size)]
def gather_rep(step, children_indices, rep_array):
    children = tf.squeeze(tf.gather(children_indices, step, axis=1))
    rep_entries = rep_array.gather(children)
    t_rep_entries = tf.transpose(rep_entries, perm=[0, 2, 1])
    return tf.transpose(tf.gather_nd(t_rep_entries, batch_indices))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
i = 0

childrens = tf.squeeze(tf.gather(children, i, axis=1))
print(sess.run(childrens))
entries = rep_array.gather(childrens)
print(sess.run(entries))

batch_indices = [[[0, 0, 0], [1, 0, 1], [2, 0, 2]],
                 [[0, 1, 0], [1, 1, 1], [2, 1, 2]],
                 [[0, 2, 0], [1, 2, 1], [2, 2, 2]],
                 [[0, 3, 0], [1, 3, 1], [2, 3, 2]]]

reps = tf.gather_nd(rep_array.gather(childrens), batch_indices)
print(sess.run(reps))

rep_entries = rep_array.gather(childrens)
print(sess.run(rep_entries))
print(sess.run(tf.shape(rep_entries)))
t_rep_entries = tf.transpose(rep_entries, perm=[0,2,1])
print(sess.run(t_rep_entries))


slice_rep_entries = tf.slice(rep_entries)
print(sess.run(slice_rep_entries))

batch_indices = list(range(3))
reps = tf.transpose(tf.gather_nd(t_rep_entries, [[0,0],[1,1],[2,2]]))
print(sess.run(reps))


reps = gather_rep(0, children, rep_array)
rep_array = rep_array.write(3, reps)
print(sess.run(tf.shape(rep_array.read(3))))
print(sess.run(rep_array.read(1)))
print(sess.run(rep_0))


t = tf.constant([[[1, 1, 1],
                  [2, 2, 2]],
                 [[3, 3, 3],
                  [4, 4, 4]],
                 [[5, 5, 5],
                  [6, 6, 6]]])
print(sess.run(tf.slice(t, [0, 1, 2],[1,1,1])))