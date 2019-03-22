import tensorflow as tf
import time

from trainers.selector import Selector
from utils import tree_util
from utils.flags import FLAGS
import utils.helper as helper
import numpy as np
from utils.summary import summarizer
from tqdm import tqdm


def train(model, load=False, gpu=True, batch_size=FLAGS.batch_size, epochs=FLAGS.epochs, run_times=[],
          epoch_times=[], conv_cond=FLAGS.conv_cond, backoff_rate=FLAGS.backoff_rate, num_threads=FLAGS.num_threads):
    helper._print_header("Training " + model.model_name)
    helper._print("Model:", model.__class__.__name__)
    helper._print("Use GPU:", gpu)
    helper._print("Test ration:", tree_util.ratio_of_labels(model.data.test_trees))
    helper._print("Validation ration:", tree_util.ratio_of_labels(model.data.val_trees))
    helper._print("Train ration:", tree_util.ratio_of_labels(model.data.train_trees))
    helper._print("Batch size:", batch_size)
    helper._print("Max epochs:", epochs)

    if gpu:
        config = tf.ConfigProto(intra_op_parallelism_threads=num_threads)
    else:
        config = tf.ConfigProto(
            device_count={'GPU': 0},
            intra_op_parallelism_threads=num_threads
        )

    conv_count = conv_cond
    with tf.Session(config=config) as sess:

        selector = Selector(model, sess, FLAGS.num_clusters, FLAGS.cluster_model)

        saver = tf.train.Saver()
        summary = summarizer(model.model_name, sess)
        summary.construct_dir()

        if load:
            model.load(sess, saver)
            summary.load()
        else:
            model.initialize(sess)
            summary.initialize()

        summary.save_parameters(lr=model.learning_rate, lr_end=model.learning_rate_end, gpu=gpu,
                                lr_decay=model.lr_decay, conv_cond=conv_cond, model=model.__class__.__name__,
                                number_variables=model.get_no_trainable_variables(),
                                max_epochs=epochs, optimizer=model.optimizer)

        epoch = summary.speed["epochs"]
        best_epoch = summary.speed["best_epoch"]
        total_time = summary.speed["total_time"]
        total_time_start = time.time()
        train_trees = model.data.train_trees
        pretrain_count = 0
        main_training = not FLAGS.use_selective_training

        while conv_count > 0 and (epochs == 0 or epochs > epoch):
            epoch += 1
            helper._print_subheader(f'Epoch {epoch} ({"Pre-training" if not main_training else "Main training"})')
            helper._print("Learning rate:", sess.run(model.lr))
            if main_training:
                helper._print(
                    f'Using {len(train_trees)}/{len(model.data.train_trees)} ({len(train_trees)/len(model.data.train_trees)*100}%) for training data.')
            start_time = time.time()
            run_time = 0
            batches = helper.batches(train_trees, batch_size, perm=True)
            pbar = tqdm(bar_format="{percentage:.0f}%|{bar}{r_bar}", total=len(batches))
            for step, batch in enumerate(batches):
                feed_dict, _ = model.build_feed_dict(batch)
                start_run_time = time.time()
                _, acc, loss = sess.run([model.train_op, model.acc, model.loss], feed_dict=feed_dict)
                end_run_time = time.time()
                run_time += end_run_time - start_run_time
                summary.add(summary.TRAIN, acc, loss)
                pbar.update(1)
            pbar.close()
            print()

            helper._print("Computing accuracies...")
            if summary.write_and_reset(summary.TRAIN, epoch, _print=True):  # training went okay
                if not main_training:
                    if summary.new_best_acc(summary.TRAIN):
                        helper._print("New best model found!")
                        pretrain_count = 0
                        model.save(sess, saver)
                        best_epoch = epoch
                    else:
                        pretrain_count += 1
                        helper._print(f"No new best model found for {pretrain_count}/{FLAGS.pretrain_stop_count} epochs. Prev best acc: {summary.best_acc[summary.TRAIN]}")
                        if pretrain_count >= FLAGS.pretrain_stop_count:
                            helper._print_header(f'PRETRAINING ENDED! Clustering for MAIN TRAINING!')
                            model.load(sess, saver)
                            train_trees = selector.select_data(model.data.train_trees, FLAGS.selection_cut_off)
                            main_training = True
                            epoch = best_epoch

                if epoch % FLAGS.val_freq == 0 and main_training:
                    summary.compute(summary.VAL, data=model.data.val_trees, model=model, epoch=epoch, _print=True)
                    # summary.compute(summary.TEST, data=model.data.test_trees, model=model, epoch=epoch, _print=True)

                    if summary.new_best_acc(summary.VAL):
                        helper._print("New best model found!!!")
                        model.save(sess, saver)
                        best_epoch = epoch
                        total_time_end = time.time()
                        total_time += total_time_end - total_time_start
                    else:
                        helper._print("No new best model found!!! Prev best loss:", summary.best_loss[summary.VAL])

                    if summary.new_best_loss(summary.TRAIN):
                        conv_count = conv_cond
                    else:
                        conv_count -= 1
                        if backoff_rate != 0 and conv_count % backoff_rate == 0:
                            helper._print("Stepping back...")
                            model.load(sess, saver)

                summary.save_speed(best_epoch, epoch, total_time)
                end_time = time.time()
                epoch_time = end_time - start_time
                epoch_times.append(epoch_time)
                helper._print("Epoch time:", str(int(epoch_time / 60)) + "m " + str(int(epoch_time % 60)) + "s")
                helper._print("Running time:", str(int(run_time / 60)) + "m " + str(int(run_time % 60)) + "s")
                if main_training:
                    helper._print("Epochs to convergence:", conv_count, "of", conv_cond)
                run_times.append(run_time)

                summary.save_history(epoch_times, run_times)
            else:
                helper._print("Nan loss was encountered, stepping back...")
                model.load(sess, saver)
                # todo maybe just one step back?

        summary.save_performance(model.data.test_trees, model)
        summary.print_performance()
        helper._print_header("Final stats for best model")
        helper._print("Total epochs:", best_epoch)
        helper._print("Total running time:",
                      str(int((total_time) / (60 * 60))) + "h",
                      str((int((total_time) / 60) % 60)) + "m")

        helper._print_subheader("Best model")
        best_step = np.argmax(np.array(summary.history[summary.VAL])[:, 1])
        helper._print_subheader("Accuracy")
        helper._print("Test:", summary.history[summary.TEST][best_step][1])
        helper._print("Validation:", summary.history[summary.VAL][best_step][1])
        helper._print("Training:", summary.history[summary.TRAIN][best_step][1])
        helper._print_subheader("Loss")
        helper._print("Test:", summary.history[summary.TEST][best_step][2])
        helper._print("Validation:", summary.history[summary.VAL][best_step][2])
        helper._print("Training:", summary.history[summary.TRAIN][best_step][2])

        helper._print_subheader("Stats")
        helper._print(summary.performance)

        summary.close()
