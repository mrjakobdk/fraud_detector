import tensorflow as tf
import time

from utils.flags import FLAGS
import utils.helper as helper
import numpy as np
from utils.summary import summarizer
from tqdm import tqdm


def train(model, load=False):
    helper._print_header("Training " + FLAGS.model_name[:-1])
    sess = tf.Session()
    saver = tf.train.Saver()

    summary = summarizer(FLAGS.model_name, sess)

    if load:
        model.load(sess, saver)
        summary.load()
    else:
        model.initialize(sess)
        summary.initialize()

    for epoch in range(1, FLAGS.epochs + 1):
        helper._print_subheader("Epoch " + str(epoch))
        helper._print("Learning rate:", sess.run(model.learning_rate))
        start_time = time.time()

        batches = helper.batches(model.data.train_trees, FLAGS.batch_size)
        pbar = tqdm(bar_format="{percentage:.0f}%|{bar}{r_bar}", total=len(batches))
        for step, batch in enumerate(batches):
            feed_dict = model.build_feed_dict(batch)
            _, acc, loss = sess.run([model.train_op, model.acc, model.loss], feed_dict=feed_dict)
            summary.add(summary.TRAIN, acc, loss)
            pbar.update(1)
        pbar.close()
        print()

        end_time = time.time()
        diff_time = end_time - start_time

        helper._print("Computing accuracies...")
        summary.write_and_reset(summary.TRAIN, epoch, _print=True)
        summary.compute(summary.VAL, data=model.data.val_trees, model=model, sess=sess, epoch=epoch, _print=True)
        summary.compute(summary.TEST, data=model.data.test_trees, model=model, sess=sess, epoch=epoch, _print=True)

        summary.save_all()

        if summary.new_best(summary.VAL):
            model.save(sess, saver)

        helper._print("Epoch time:", str(int(diff_time / 60)) + "m " + str(int(diff_time % 60))+"s")
