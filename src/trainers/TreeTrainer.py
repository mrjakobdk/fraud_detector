import tensorflow as tf
import time

from trainers.selector import Selector
from utils import tree_util, constants
from utils.flags import FLAGS
import utils.helper as helper
import numpy as np
from utils.summary import summarizer
from tqdm import tqdm
import math


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
        helper._print("Drop epochs:", FLAGS.pretrain_max_epoch)

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
            pbar = tqdm(bar_format="{percentage:.0f}%|{bar}{r_bar}", total=len(batches))
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
        config = tf.ConfigProto(intra_op_parallelism_threads=num_threads)
    else:
        config = tf.ConfigProto(
            device_count={'GPU': 0},
            intra_op_parallelism_threads=num_threads
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
            #summary.dropping_tick()
            summary.save_speed()
            summary.pre_tick()

        # todo maybe allow multiple repeat selective training
        # Selecting
        helper._print_header('PRETRAINING ENDED!')
        model.load_best(sess, saver, summary.TRAIN)

        # Main training
        main_count = 0
        while not summary.converging() and not summary.interrupt():
            main_count += 1
            if main_count == 1 or (FLAGS.use_multi_cluster and main_count % int(FLAGS.pretrain_max_epoch/4)==0):
                helper._print_header(f'Clustering for MAIN TRAINING!')
                train_data_selection = selector.select_data(model.data.train_trees, FLAGS.selection_cut_off)
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
                helper._print("No new best model found!!! Prev best validation acc:", summary.speed["converging_acc"])
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

# def train_old(model, load=False, gpu=True, batch_size=FLAGS.batch_size, epochs=FLAGS.epochs, run_times=[],
#               epoch_times=[], conv_cond=FLAGS.conv_cond, backoff_rate=FLAGS.backoff_rate,
#               num_threads=FLAGS.num_threads):
#     helper._print_header("Training " + model.model_name)
#     helper._print("Model:", model.__class__.__name__)
#     helper._print("Use GPU:", gpu)
#     helper._print("Test ration:", tree_util.ratio_of_labels(model.data.test_trees))
#     helper._print("Validation ration:", tree_util.ratio_of_labels(model.data.val_trees))
#     helper._print("Train ration:", tree_util.ratio_of_labels(model.data.train_trees))
#     helper._print("Batch size:", batch_size)
#     helper._print("Max epochs:", epochs)
#
#     if gpu:
#         config = tf.ConfigProto(intra_op_parallelism_threads=num_threads)
#     else:
#         config = tf.ConfigProto(
#             device_count={'GPU': 0},
#             intra_op_parallelism_threads=num_threads
#         )
#
#     conv_count = conv_cond
#     with tf.Session(config=config) as sess:
#
#         selector = Selector(model, sess, FLAGS.num_clusters, FLAGS.cluster_model)
#
#         saver = tf.train.Saver()
#         summary = summarizer(model.model_name, sess)
#         summary.construct_dir()
#
#         if load:
#             model.load(sess, saver)
#             summary.load()
#         else:
#             model.initialize(sess)
#             summary.initialize()
#
#         summary.save_parameters(lr=model.learning_rate, lr_end=model.learning_rate_end, gpu=gpu,
#                                 lr_decay=model.lr_decay, conv_cond=conv_cond, model=model.__class__.__name__,
#                                 number_variables=model.get_no_trainable_variables(),
#                                 max_epochs=epochs, optimizer=model.optimizer)
#
#         epoch = summary.speed["epochs"]
#         best_epoch = summary.speed["best_epoch"]
#         total_time = summary.speed["total_time"]
#         total_time_start = time.time()
#         train_trees = model.data.train_trees
#         pretrain_count = 0
#         main_training = not FLAGS.use_selective_training
#         prev_best_train_acc = summary.best_acc[summary.TRAIN]
#         prev_best_val_acc = summary.best_acc[summary.VAL]
#
#         while conv_count > 0 and (epochs == 0 or epochs > epoch):
#             epoch += 1
#             helper._print_subheader(f'Epoch {epoch} ({"Pre-training" if not main_training else "Main training"})')
#             helper._print("Learning rate:", sess.run(model.lr))
#             if main_training:
#                 helper._print(
#                     f'Using {len(train_trees)}/{len(model.data.train_trees)} ({len(train_trees)/len(model.data.train_trees)*100}%) for training data.')
#             start_time = time.time()
#             run_time = 0
#
#             batches = helper.batches(train_trees, batch_size, perm=True)
#             pbar = tqdm(bar_format="{percentage:.0f}%|{bar}{r_bar}", total=len(batches))
#             for step, batch in enumerate(batches):
#                 feed_dict, _ = model.build_feed_dict(batch)
#                 start_run_time = time.time()
#                 _, acc, loss = sess.run([model.train_op, model.acc, model.loss], feed_dict=feed_dict)
#                 end_run_time = time.time()
#                 run_time += end_run_time - start_run_time
#                 summary.add(summary.TRAIN, acc, loss)
#                 pbar.update(1)
#             pbar.close()
#             print()
#
#             helper._print("Computing accuracies...")
#             if summary.write_and_reset(summary.TRAIN, epoch, _print=True):  # training went okay
#                 if not main_training:
#                     if summary.new_best_acc(summary.TRAIN):
#                         helper._print("New best model found!")
#                         if summary.best_acc[summary.TRAIN] - prev_best_train_acc > FLAGS.acc_min_delta_drop:
#                             pretrain_count = 0
#                             prev_best_train_acc = summary.best_acc[summary.TRAIN]
#                         else:
#                             pretrain_count += 1
#                         model.save(sess, saver)
#                         best_epoch = epoch
#                     else:
#                         pretrain_count += 1
#                         helper._print(
#                             f"No new best model found for {pretrain_count}/{FLAGS.pretrain_stop_count} epochs. Prev best acc: {summary.best_acc[summary.TRAIN]}")
#                         if pretrain_count >= FLAGS.pretrain_stop_count:
#                             helper._print_header(f'PRETRAINING ENDED! Clustering for MAIN TRAINING!')
#                             model.load(sess, saver)
#                             train_trees = selector.select_data(model.data.train_trees, FLAGS.selection_cut_off)
#                             main_training = True
#                             epoch = best_epoch
#
#                 if epoch % FLAGS.val_freq == 0 and main_training:
#                     summary.compute(summary.VAL, data=model.data.val_trees, model=model, epoch=epoch, _print=True)
#                     # summary.compute(summary.TEST, data=model.data.test_trees, model=model, epoch=epoch, _print=True)
#
#                     if summary.new_best_acc(summary.VAL):
#                         helper._print("New best model found!!!")
#                         model.save(sess, saver)
#                         best_epoch = epoch
#                         total_time_end = time.time()
#                         total_time += total_time_end - total_time_start
#                     else:
#                         helper._print("No new best model found!!! Prev best acc:", summary.best_acc[summary.VAL])
#
#                     if summary.new_best_acc(summary.VAL):
#                         if summary.best_acc[summary.VAL] - prev_best_val_acc > FLAGS.acc_min_delta_conv:
#                             conv_count = conv_cond
#                             prev_best_val_acc = summary.best_acc[summary.VAL]
#                         else:
#                             conv_count -= 1
#                     else:
#                         conv_count -= 1
#                         if backoff_rate != 0 and conv_count % backoff_rate == 0:
#                             helper._print("Stepping back...")
#                             model.load(sess, saver)
#
#                 summary.save_speed(best_epoch, epoch, total_time)
#                 end_time = time.time()
#                 epoch_time = end_time - start_time
#                 epoch_times.append(epoch_time)
#                 helper._print("Epoch time:", str(int(epoch_time / 60)) + "m " + str(int(epoch_time % 60)) + "s")
#                 helper._print("Running time:", str(int(run_time / 60)) + "m " + str(int(run_time % 60)) + "s")
#                 if main_training:
#                     helper._print("Epochs to convergence:", conv_count, "of", conv_cond)
#                 run_times.append(run_time)
#
#                 summary.save_history(epoch_times, run_times)
#             else:
#                 helper._print("Nan loss was encountered, stepping back...")
#                 model.load(sess, saver)
#                 # todo maybe just one step back?
#
#         summary.save_performance(model.data, model)
#         summary.print_performance()
#         # helper._print_header("Final stats for best model")
#         # helper._print("Total epochs:", best_epoch)
#         # helper._print("Total running time:",
#         #               str(int((total_time) / (60 * 60))) + "h",
#         #               str((int((total_time) / 60) % 60)) + "m")
#         #
#         # helper._print_subheader("Best model")
#         # best_step = np.argmax(np.array(summary.history[summary.VAL])[:, 1])
#         # helper._print_subheader("Accuracy")
#         # helper._print("Test:", summary.history[summary.TEST][best_step][1])
#         # helper._print("Validation:", summary.history[summary.VAL][best_step][1])
#         # helper._print("Training:", summary.history[summary.TRAIN][best_step][1])
#         # helper._print_subheader("Loss")
#         # helper._print("Test:", summary.history[summary.TEST][best_step][2])
#         # helper._print("Validation:", summary.history[summary.VAL][best_step][2])
#         # helper._print("Training:", summary.history[summary.TRAIN][best_step][2])
#         #
#         # helper._print_subheader("Stats")
#         # helper._print(summary.performance)
#
#         summary.close()
