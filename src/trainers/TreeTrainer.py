import tensorflow as tf
import time
import utils.helper as helper

from trainers.selector import Selector
from utils import tree_util
from utils.flags import FLAGS
from utils.summary import summarizer
from tqdm import tqdm

class Trainer():
    def __init__(self, model, sess, saver, summary, load=False, gpu=True, batch_size=FLAGS.batch_size,
                 epochs=FLAGS.epochs):
        helper._print_header("Training " + model.model_name)
        helper._print("Load model:", load)
        helper._print("Model:", model.__class__.__name__)
        helper._print("Use GPU:", gpu)
        helper._print("Test ration:", tree_util.ratio_of_labels(model.data.test_trees))
        helper._print("Validation ration:", tree_util.ratio_of_labels(model.data.val_trees))
        helper._print("Train ration:", tree_util.ratio_of_labels(model.data.train_trees))
        helper._print("Batch size:", batch_size)
        helper._print("Max epochs:", epochs)
        helper._print("Convergence epochs:", FLAGS.conv_cond)
        helper._print("Max pre-training epochs:", FLAGS.pretrain_max_epoch)

        self.model = model
        self.batch_size = batch_size
        self.sess = sess
        self.saver = saver
        self.summary = summary

    def train(self, train_data):
        helper._print("Learning rate:", self.sess.run(self.model.lr))
        done = False
        while not done:
            run_time = 0
            batches = helper.batches(train_data, self.batch_size, perm=True)
            pbar = tqdm(
                bar_format="{percentage:.0f}%|{bar}| Elapsed: {elapsed}, Remaining: {remaining} ({n_fmt}/{total_fmt})",
                total=len(batches))
            for step, batch in enumerate(batches):
                self.summary.batch_inc()
                feed_dict, _ = self.model.build_feed_dict(batch)
                start_run_time = time.time()
                _, acc, loss = self.sess.run([self.model.train_op, self.model.acc, self.model.loss],
                                             feed_dict=feed_dict)
                end_run_time = time.time()
                run_time += end_run_time - start_run_time
                self.summary.add(self.summary.TRAIN, acc, loss)
                pbar.update(1)
            pbar.close()
            print()

            # loading and saving tmp model - just in case something goes wrong
            if not self.summary.write_and_reset(self.summary.TRAIN, _print=True):
                helper._print("Nan loss encountered, trying again...")
                self.model.load_tmp(self.sess, self.saver)
            else:
                done = True
                self.model.save_tmp(self.sess, self.saver)

            helper._print("Training time:", str(int(run_time / 60)) + "m " + str(int(run_time % 60)) + "s")


def selective_train(model, load=False, gpu=True, batch_size=FLAGS.batch_size, epochs=FLAGS.epochs, run_times=[],
                    epoch_times=[], conv_cond=FLAGS.conv_cond,
                    num_threads=FLAGS.num_threads):
    if gpu:
        config = None
    else:
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )

    with tf.Session(config=config) as sess:

        # initialization
        saver = tf.train.Saver()
        selector = Selector(model, sess, FLAGS.num_clusters, FLAGS.cluster_model)
        summary = summarizer(model.model_name, sess)
        summary.construct_dir()
        trainer = Trainer(model, sess, saver, summary, load=load, gpu=gpu, batch_size=batch_size)

        if load:
            model.load_tmp(sess, saver)
            summary.load()
        else:
            model.initialize(sess)
            summary.initialize()

        summary.save_parameters(lr=model.learning_rate, lr_end=model.learning_rate_end, gpu=gpu,
                                lr_decay=model.lr_decay, conv_cond=conv_cond, model=model.__class__.__name__,
                                number_variables=model.get_no_trainable_variables(),
                                max_epochs=epochs, optimizer=model.optimizer)

        # Pre-training
        train_data, val_data, test_data = model.data.train_trees, model.data.val_trees, model.data.test_trees
        while not summary.dropping() and not summary.interrupt():
            summary.epoch_inc()
            helper._print_subheader(f'Epoch {summary.get_epoch()} (Pre-training)')

            trainer.train(train_data)

            summary.compute(summary.VAL, data=model.data.val_trees, model=model, _print=True)

            summary.save_history()
            summary.time_tick()

            if summary.new_best_acc(summary.VAL):
                helper._print("New best val model found!")
                model.save_best(sess, saver, summary.VAL)

            if summary.new_best_acc(summary.TRAIN):
                helper._print("New best train model found!")
                model.save_best(sess, saver, summary.TRAIN)
            else:
                helper._print("No new best model found!!! Prev best training acc:", summary.best_acc[summary.TRAIN])
            # summary.dropping_tick()
            summary.save_speed()
            summary.pre_tick()

        # todo maybe allow multiple repeat selective training
        # Selecting
        helper._print_header('PRETRAINING ENDED!')
        model.load_best(sess, saver, summary.TRAIN)

        # Main training
        first = True
        while not summary.converging() and not summary.interrupt():
            summary.main_count_tick()
            if first and FLAGS.load_model:
                cluster_predictions = summary.load_cluster_predictions()
                train_data_selection, cluster_predictions = selector.select_data(model.data.train_trees,
                                                                                 FLAGS.selection_cut_off,
                                                                                 cluster_predictions=cluster_predictions)
                first = False

            if summary.re_cluster():
                # if main_count == 1 or (FLAGS.use_multi_cluster and main_count % int(FLAGS.pretrain_max_epoch/4)==0):
                helper._print_header(f'Clustering for MAIN TRAINING!')
                train_data_selection, cluster_predictions = selector.select_data(model.data.train_trees,
                                                                                 FLAGS.selection_cut_off)
                summary.save_cluster_predictions(cluster_predictions)
                summary.time_tick("Selection time:")

            summary.epoch_inc()

            helper._print_subheader(f'Epoch {summary.get_epoch()} (Main training)')
            helper._print(
                f'Using {len(train_data_selection)}/{len(train_data)} ({len(train_data_selection)/len(train_data)*100}%) for training data.')

            trainer.train(train_data_selection)

            summary.compute(summary.VAL, data=model.data.val_trees, model=model, _print=True)
            summary.save_history()
            summary.time_tick()

            if summary.new_best_acc(summary.VAL):
                helper._print("New best model found!")
                model.save_best(sess, saver, summary.VAL)
            else:
                helper._print("No new best model found!!! Prev best validation acc:", summary.best_acc[summary.VAL])
            summary.converging_tick()
            summary.save_speed()

        model.load_best(sess, saver, summary.VAL)
        summary.save_performance(model)
        summary.print_performance()


def train(model, load=False, gpu=True, batch_size=FLAGS.batch_size, epochs=FLAGS.epochs, run_times=[],
          epoch_times=[], conv_cond=FLAGS.conv_cond,
          num_threads=FLAGS.num_threads):
    if gpu:
        config = tf.ConfigProto(intra_op_parallelism_threads=num_threads)
    else:
        config = tf.ConfigProto(
            device_count={'GPU': 0},
            intra_op_parallelism_threads=num_threads
        )

    with tf.Session(config=config) as sess:
        # initialization
        saver = tf.train.Saver()
        summary = summarizer(model.model_name, sess)
        summary.construct_dir()
        trainer = Trainer(model, sess, saver, summary, load=load, gpu=gpu, batch_size=batch_size)

        if load:
            model.load_tmp(sess, saver)
            summary.load()
        else:
            model.initialize(sess)
            summary.initialize()

        summary.save_parameters(lr=model.learning_rate, lr_end=model.learning_rate_end, gpu=gpu,
                                lr_decay=model.lr_decay, conv_cond=conv_cond, model=model.__class__.__name__,
                                number_variables=model.get_no_trainable_variables(),
                                max_epochs=epochs, optimizer=model.optimizer)

        # Training
        train_data, val_data, test_data = model.data.train_trees, model.data.val_trees, model.data.test_trees
        while not summary.converging() and not summary.at_max_epoch() and not summary.interrupt():
            summary.epoch_inc()
            helper._print_subheader(f'Epoch {summary.get_epoch()}')

            run_time = trainer.train(train_data)
            run_times.append(run_time)

            summary.compute(summary.VAL, data=model.data.val_trees, model=model, _print=True)
            summary.save_history()
            summary.time_tick()
            epoch_times.append(summary.get_time())
            if summary.new_best_acc(summary.VAL):
                helper._print("New best model found!")
                model.save_best(sess, saver)
            else:
                helper._print("No new best model found!!! Prev best validation acc:", summary.best_acc[summary.VAL])
            summary.converging_tick()
            summary.save_speed()

            model.load_best(sess, saver)
            summary.save_performance(model)
            summary.print_performance()