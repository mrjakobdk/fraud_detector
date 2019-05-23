import os
import numpy as np
import pandas as pd

from utils import helper


def get_models_info(models, batch_sizes, lr_decay, model_names):
    acc_val = []
    acc_test = []
    for model_name in model_names:
        val = helper.load_dict(f"../trained_models/{model_name}/performance_val.csv")
        test = helper.load_dict(f"../trained_models/{model_name}/performance_test.csv")
        acc_val.append(round(val["accuracy"], 4))
        acc_test.append(round(test["accuracy"], 4))

    return {
        "Model": models,
        "Batch size": batch_sizes,
        "Lr decay": lr_decay,
        "acc val": acc_val,
        "acc test": acc_test,
    }


def get_acc_table_batch_size_and_lrdecay():
    treernn = get_models_info(models=["TreeRNN"] * 4,
                              batch_sizes=[4, 4, 64, 64],
                              lr_decay=[0.995, 0.98, 0.995, 0.98],
                              model_names=["treeRNN_neerbek_batch4_decay995_rep100",
                                           "treeRNN_neerbek_batch4_decay98_rep100",
                                           "treeRNN_neerbek_batch64_decay995_rep100",
                                           "treeRNN_neerbek_batch64_decay98_rep100"])

    mtreernn = get_models_info(models=["MTreeRNN"] * 4,
                               batch_sizes=[4, 4, 64, 64],
                               lr_decay=[0.995, 0.98, 0.995, 0.98],
                               model_names=["treeRNN_batch_batch4_decay995_rep100",
                                            "treeRNN_batch_batch4_decay98_rep100",
                                            "treeRNN_batch_batch64_decay995_rep100",
                                            "treeRNN_batch_batch64_decay98_rep100"])

    deepRNN = get_models_info(models=["DeepRNN"] * 4,
                              batch_sizes=[4, 4, 64, 64],
                              lr_decay=[0.995, 0.98, 0.995, 0.98],
                              model_names=["deepRNN_batch4_decay995_rep100",
                                           "deepRNN_batch4_decay98_rep100",
                                           "deepRNN_batch64_decay995_rep100",
                                           "deepRNN_batch64_decay98_rep100"])

    treeLSTM_no_cross = get_models_info(models=["TreeLSTM WRONG"] * 3,
                                        batch_sizes=[4, 64, 64],
                                        lr_decay=[0.995, 0.995, 0.98],
                                        model_names=["WRONG_treeLSTM_batch4_decay995_rep100",
                                                     "WRONG_treeLSTM_batch64_decay995_rep100",
                                                     "WRONG_treeLSTM_batch64_decay98_rep100"])

    df = pd.DataFrame(treernn)
    df = df.append(pd.DataFrame(mtreernn))
    df = df.append(pd.DataFrame(deepRNN))
    df = df.append(pd.DataFrame(treeLSTM_no_cross))

    print(df.to_latex(columns=["Model", "Batch size", "Lr decay", "acc val", "acc test"], index=False))


def get_acc_table_deep_layers():
    deepRNN = get_models_info(
        models=["DeepRNN layers 1", "DeepRNN layers 2", "DeepRNN layers 3", "DeepRNN layers 4", "DeepRNN layers 5"],
        batch_sizes=[4] * 5,
        lr_decay=[0.980] * 5,
        model_names=["treeRNN_neerbek_batch4_decay98_rep100",
                     "deepRNN_batch4_decay98_rep100_layers2",
                     "deepRNN_batch4_decay98_rep100",
                     "deepRNN_batch4_decay98_rep100_layers4",
                     "deepRNN_batch4_decay98_rep100_layers5"])
    df = pd.DataFrame(deepRNN)
    print(df.to_latex(columns=["Model", "Batch size", "Lr decay", "acc val", "acc test"], index=False))


def get_Exp1_info(models, batch_sizes, model_names):
    acc_val = []
    acc_test = []
    models_list = []
    batch_size_list = []
    for model_name, batch_size, model in zip(model_names, batch_sizes, models):
        if os.path.exists(f"../trained_models/{model_name}"):
            val = helper.load_dict(f"../trained_models/{model_name}/performance_val.csv")
            test = helper.load_dict(f"../trained_models/{model_name}/performance_test.csv")
            acc_val.append(round(val["accuracy"], 4))
            acc_test.append(round(test["accuracy"], 4))
            batch_size_list.append(batch_size)
            models_list.append(model)
        else:
            print(model_name, "does not exists")

    return {
        "Model": models_list,
        "Batch size": batch_size_list,
        "acc val": acc_val,
    }


def get_Exp1():
    treernn = get_Exp1_info(models=["TreeRNN"] * 3,
                            batch_sizes=[4, 16, 64],
                            model_names=["TreeRNN_batch4_decay98_rep100_lr1_normal_train_conv100_adagrad",
                                         "TreeRNN_batch16_decay98_rep100_lr1_normal_train_conv100_adagrad",
                                         "TreeRNN_batch64_decay98_rep100_lr1_normal_train_conv100_adagrad"])

    mtreernn = get_Exp1_info(models=["MTreeRNN"] * 4,
                             batch_sizes=[4, 16, 64],
                             model_names=["MTreeRNN_batch4_decay98_rep100_lr1_normal_train_conv100_adagrad",
                                          "MTreeRNN_batch16_decay98_rep100_lr1_normal_train_conv100_adagrad",
                                          "MTreeRNN_batch64_decay98_rep100_lr1_normal_train_conv100_adagrad"])

    deepRNN = get_Exp1_info(models=["DeepRNN"] * 4,
                            batch_sizes=[4, 16, 64],
                            model_names=["DeepRNN_batch4_decay98_rep100_lr1_normal_train_conv100_adagrad",
                                         "DeepRNN_batch16_decay98_rep100_lr1_normal_train_conv100_adagrad",
                                         "DeepRNN_batch64_decay98_rep100_lr1_normal_train_conv100_adagrad"])

    treeLSTM = get_Exp1_info(models=["TreeLSTM"] * 3,
                             batch_sizes=[4, 16, 64],
                             model_names=["TreeLSTM_batch4_decay98_rep100_lr1_normal_train_conv100_adagrad",
                                          "TreeLSTM_batch16_decay98_rep100_lr1_normal_train_conv100_adagrad",
                                          "TreeLSTM_batch64_decay98_rep100_lr1_normal_train_conv100_adagrad"])

    treeLSTM_tacker = get_Exp1_info(models=["TreeLSTM with Tracker"] * 3,
                                    batch_sizes=[4, 16, 64],
                                    model_names=["Tracker_batch4_decay98_rep100_lr1_normal_train_conv100_adagrad",
                                                 "Tracker_batch16_decay98_rep100_lr1_normal_train_conv100_adagrad",
                                                 "Tracker_batch64_decay98_rep100_lr1_normal_train_conv100_adagrad"])

    LSTM = get_Exp1_info(models=["LSTM"] * 3,
                         batch_sizes=[4, 16, 64],
                         model_names=["LSTM_batch4_decay98_rep100_lr1_normal_train_conv100_adagrad",
                                      "LSTM_batch16_decay98_rep100_lr1_normal_train_conv100_adagrad",
                                      "LSTM_batch64_decay98_rep100_lr1_normal_train_conv100_adagrad"])

    df = pd.DataFrame(treernn)
    df = df.append(pd.DataFrame(mtreernn))
    df = df.append(pd.DataFrame(deepRNN))
    df = df.append(pd.DataFrame(treeLSTM))
    df = df.append(pd.DataFrame(treeLSTM_tacker))
    df = df.append(pd.DataFrame(LSTM))

    print(df.to_latex(columns=["Model", "Batch size", "acc val"], index=False))


def get_Exp2_info(models, lrs, lr_decays, model_names):
    acc_train = []
    acc_val = []
    acc_test = []
    models_list = []
    lr_list = []
    lr_decay_list = []
    for model_name, lr, lr_decays, model in zip(model_names, lrs, lr_decays, models):
        if os.path.exists(f"../trained_models/{model_name}"):
            if os.path.exists(f"../trained_models/{model_name}/performance_train.csv"):
                train = helper.load_dict(f"../trained_models/{model_name}/performance_train.csv")
                val = helper.load_dict(f"../trained_models/{model_name}/performance_val.csv")
                test = helper.load_dict(f"../trained_models/{model_name}/performance_test.csv")
                acc_train.append(round(train["accuracy"], 4))
                acc_val.append(round(val["accuracy"], 4))
                acc_test.append(round(test["accuracy"], 4))
                lr_list.append(lr)
                lr_decay_list.append(lr_decays)
                models_list.append(model)
            else:
                print(model_name, "does not have performance file")
        else:
            print(model_name, "does not exists")

    return {
        "Model": models_list,
        "Learning rate": lr_list,
        "Learning rate decay": lr_decay_list,
        "Train Acc": acc_train,
        "Val Acc": acc_val,
        "Test Acc": acc_test,
    }


def get_Exp2():
    treernn = get_Exp2_info(models=["TreeRNN"] * 6,
                            lrs=[0.1, 0.1, 0.1, 0.01, 0.01, 0.01],
                            lr_decays=[0.98, 0.995, 1, 0.98, 0.995, 1],
                            model_names=["TreeRNN_batch4_decay98_rep100_lr1_normal_train_conv100_adagrad",
                                         "TreeRNN_batch4_decay995_rep100_lr1_normal_train_conv100_adagrad",
                                         "TreeRNN_batch4_decay1_rep100_lr1_normal_train_conv100_adagrad",
                                         "TreeRNN_batch4_decay98_rep100_lr01_normal_train_conv100_adagrad",
                                         "TreeRNN_batch4_decay995_rep100_lr01_normal_train_conv100_adagrad",
                                         "TreeRNN_batch4_decay1_rep100_lr01_normal_train_conv100_adagrad"])

    mtreernn = get_Exp2_info(models=["MTreeRNN"] * 6,
                             lrs=[0.1, 0.1, 0.1, 0.01, 0.01, 0.01],
                             lr_decays=[0.98, 0.995, 1, 0.98, 0.995, 1],
                             model_names=["MTreeRNN_batch16_decay98_rep100_lr1_normal_train_conv100_adagrad",
                                          "MTreeRNN_batch16_decay995_rep100_lr1_normal_train_conv100_adagrad",
                                          "MTreeRNN_batch16_decay1_rep100_lr1_normal_train_conv100_adagrad",
                                          "MTreeRNN_batch16_decay98_rep100_lr01_normal_train_conv100_adagrad",
                                          "MTreeRNN_batch16_decay995_rep100_lr01_normal_train_conv100_adagrad",
                                          "MTreeRNN_batch16_decay1_rep100_lr01_normal_train_conv100_adagrad"])

    deepRNN = get_Exp2_info(models=["DeepRNN"] * 6,
                            lrs=[0.1, 0.1, 0.1, 0.01, 0.01, 0.01],
                            lr_decays=[0.98, 0.995, 1, 0.98, 0.995, 1],
                            model_names=["DeepRNN_batch4_decay98_rep100_lr1_normal_train_conv100_adagrad",
                                         "DeepRNN_batch4_decay995_rep100_lr1_normal_train_conv100_adagrad",
                                         "DeepRNN_batch4_decay1_rep100_lr1_normal_train_conv100_adagrad",
                                         "DeepRNN_batch4_decay98_rep100_lr01_normal_train_conv100_adagrad",
                                         "DeepRNN_batch4_decay995_rep100_lr01_normal_train_conv100_adagrad",
                                         "DeepRNN_batch4_decay1_rep100_lr01_normal_train_conv100_adagrad"])

    treeLSTM = get_Exp2_info(models=["TreeLSTM"] * 6,
                             lrs=[0.1, 0.1, 0.1, 0.01, 0.01, 0.01],
                             lr_decays=[0.98, 0.995, 1, 0.98, 0.995, 1],
                             model_names=["TreeLSTM_batch4_decay98_rep100_lr1_normal_train_conv100_adagrad",
                                          "TreeLSTM_batch4_decay995_rep100_lr1_normal_train_conv100_adagrad",
                                          "TreeLSTM_batch4_decay1_rep100_lr1_normal_train_conv100_adagrad",
                                          "TreeLSTM_batch4_decay98_rep100_lr01_normal_train_conv100_adagrad",
                                          "TreeLSTM_batch4_decay995_rep100_lr01_normal_train_conv100_adagrad",
                                          "TreeLSTM_batch4_decay1_rep100_lr01_normal_train_conv100_adagrad"])

    treeLSTM_tacker = get_Exp2_info(models=["TreeLSTM with Tracker"] * 6,
                                    lrs=[0.1, 0.1, 0.1, 0.01, 0.01, 0.01],
                                    lr_decays=[0.98, 0.995, 1, 0.98, 0.995, 1],
                                    model_names=["Tracker_batch16_decay98_rep100_lr1_normal_train_conv100_adagrad",
                                                 "Tracker_batch16_decay995_rep100_lr1_normal_train_conv100_adagrad",
                                                 "Tracker_batch16_decay1_rep100_lr1_normal_train_conv100_adagrad",
                                                 "Tracker_batch16_decay98_rep100_lr01_normal_train_conv100_adagrad",
                                                 "Tracker_batch16_decay995_rep100_lr01_normal_train_conv100_adagrad",
                                                 "Tracker_batch16_decay1_rep100_lr01_normal_train_conv100_adagrad"])

    LSTM = get_Exp2_info(models=["LSTM"] * 6,
                         lrs=[0.1, 0.1, 0.1, 0.01, 0.01, 0.01],
                         lr_decays=[0.98, 0.995, 1, 0.98, 0.995, 1],
                         model_names=["LSTM_batch4_decay98_rep100_lr1_normal_train_conv100_adagrad",
                                      "LSTM_batch4_decay995_rep100_lr1_normal_train_conv100_adagrad",
                                      "LSTM_batch4_decay1_rep100_lr1_normal_train_conv100_adagrad",
                                      "LSTM_batch4_decay98_rep100_lr01_normal_train_conv100_adagrad",
                                      "LSTM_batch4_decay995_rep100_lr01_normal_train_conv100_adagrad",
                                      "LSTM_batch4_decay1_rep100_lr01_normal_train_conv100_adagrad"])

    df = pd.DataFrame(treernn)
    df = df.append(pd.DataFrame(mtreernn))
    df = df.append(pd.DataFrame(deepRNN))
    df = df.append(pd.DataFrame(treeLSTM))
    df = df.append(pd.DataFrame(treeLSTM_tacker))
    df = df.append(pd.DataFrame(LSTM))

    print(df.to_latex(columns=["Model", "Learning rate", "Learning rate decay", "Train Acc", "Val Acc", "Test Acc"],
                      index=False))


def get_Exp3_info(models, optimizers, lrs, lr_decays, model_names):
    acc_train = []
    acc_val = []
    acc_test = []
    models_list = []
    optimizer_list = []
    lr_list = []
    lr_decay_list = []
    for model_name, optimizer, lr, lr_decays, model in zip(model_names, optimizers, lrs, lr_decays, models):
        if os.path.exists(f"../trained_models/{model_name}"):
            if os.path.exists(f"../trained_models/{model_name}/performance_train.csv"):
                train = helper.load_dict(f"../trained_models/{model_name}/performance_train.csv")
                val = helper.load_dict(f"../trained_models/{model_name}/performance_val.csv")
                test = helper.load_dict(f"../trained_models/{model_name}/performance_test.csv")
                acc_train.append(round(train["accuracy"], 4))
                acc_val.append(round(val["accuracy"], 4))
                acc_test.append(round(test["accuracy"], 4))
                optimizer_list.append(optimizer)
                lr_list.append(lr)
                lr_decay_list.append(lr_decays)
                models_list.append(model)
            else:
                print(model_name, "does not have performance file")
        else:
            print(model_name, "does not exists")

    return {
        "Model": models_list,
        "Optimizer": optimizer_list,
        "Learning rate": lr_list,
        "Learning rate decay": lr_decay_list,
        "Train Acc": acc_train,
        "Val Acc": acc_val,
        "Test Acc": acc_test,
    }


def get_Exp3():
    treernn = get_Exp3_info(models=["TreeRNN"] * 3,
                            optimizers=["AdaGrad", "Adam", "Adam"],
                            lrs=[0.01, 0.001, 0.0001],
                            lr_decays=[0.995, 1, 1],
                            model_names=["TreeRNN_batch4_decay995_rep100_lr01_normal_train_conv100_adagrad",
                                         "TreeRNN_batch4_decay1_rep100_lr001_normal_train_conv100_adam",
                                         "TreeRNN_batch4_decay1_rep100_lr0001_normal_train_conv100_adam"])

    mtreernn = get_Exp3_info(models=["MTreeRNN"] * 3,
                             optimizers=["AdaGrad", "Adam", "Adam"],
                             lrs=[0.01, 0.001, 0.0001],
                             lr_decays=[1, 1, 1],
                             model_names=["MTreeRNN_batch16_decay1_rep100_lr01_normal_train_conv100_adagrad",
                                          "MTreeRNN_batch16_decay1_rep100_lr001_normal_train_conv100_adam",
                                          "MTreeRNN_batch16_decay1_rep100_lr0001_normal_train_conv100_adam"])

    deepRNN = get_Exp3_info(models=["DeepRNN"] * 3,
                            optimizers=["AdaGrad", "Adam", "Adam"],
                            lrs=[0.1, 0.001, 0.0001],
                            lr_decays=[0.98, 1, 1],
                            model_names=["DeepRNN_batch4_decay98_rep100_lr1_normal_train_conv100_adagrad",
                                         "DeepRNN_batch4_decay1_rep100_lr001_normal_train_conv100_adam",
                                         "DeepRNN_batch4_decay1_rep100_lr0001_normal_train_conv100_adam"])

    treeLSTM = get_Exp3_info(models=["TreeLSTM"] * 3,
                             optimizers=["AdaGrad", "Adam", "Adam"],
                             lrs=[0.01, 0.001, 0.0001],
                             lr_decays=[1, 1, 1],
                             model_names=["TreeLSTM_batch4_decay1_rep100_lr01_normal_train_conv100_adagrad",
                                          "TreeLSTM_batch4_decay1_rep100_lr001_normal_train_conv100_adam",
                                          "TreeLSTM_batch4_decay1_rep100_lr0001_normal_train_conv100_adam"])

    treeLSTM_tacker = get_Exp3_info(models=["TreeLSTM with Tracker"] * 3,
                                    optimizers=["AdaGrad", "Adam", "Adam"],
                                    lrs=[0.1, 0.001, 0.0001],
                                    lr_decays=[1, 1, 1],
                                    model_names=["Tracker_batch64_decay995_rep100_lr01_normal_train_conv100_adagrad",
                                                 "Tracker_batch64_decay1_rep100_lr001_normal_train_conv100_adam",
                                                 "Tracker_batch64_decay1_rep100_lr0001_normal_train_conv100_adam"])

    df = pd.DataFrame(treernn)
    df = df.append(pd.DataFrame(mtreernn))
    df = df.append(pd.DataFrame(deepRNN))
    df = df.append(pd.DataFrame(treeLSTM))
    df = df.append(pd.DataFrame(treeLSTM_tacker))

    print(df.to_latex(
        columns=["Model", "Optimizer", "Learning rate", "Learning rate decay", "Train Acc", "Val Acc"],
        index=False))


def get_Exp4_info(models, dropouts, l2s, model_names):
    acc_train = []
    acc_val = []
    acc_test = []
    models_list = []
    dropout_list = []
    l2_list = []
    for model_name, dropout, l2, model in zip(model_names, dropouts, l2s, models):
        if os.path.exists(f"../trained_models/{model_name}"):
            if os.path.exists(f"../trained_models/{model_name}/performance_train.csv"):
                train = helper.load_dict(f"../trained_models/{model_name}/performance_train.csv")
                val = helper.load_dict(f"../trained_models/{model_name}/performance_val.csv")
                test = helper.load_dict(f"../trained_models/{model_name}/performance_test.csv")
                acc_train.append(round(train["accuracy"], 4))
                acc_val.append(round(val["accuracy"], 4))
                acc_test.append(round(test["accuracy"], 4))
                dropout_list.append(dropout)
                l2_list.append(l2)
                models_list.append(model)
            else:
                print(model_name, "does not have performance file")
        else:
            print(model_name, "does not exists")

    return {
        "Model": models_list,
        "Dropout Rate": dropout_list,
        "L2 Scalar": l2_list,
        "Train Acc": acc_train,
        "Val Acc": acc_val,
        "Test Acc": acc_test,
    }


def get_Exp4():
    treernn = get_Exp4_info(models=["TreeRNN"] * 9,
                            dropouts=["0%", "0%", "0%", "0%", "10%", "25%", "50%", "5%", "10%"],
                            l2s=[0, 0.01, 0.001, 0.0001, 0, 0, 0, 0.00005, 0.0001],
                            model_names=["TreeRNN_batch4_decay995_rep100_lr01_normal_train_conv100_adagrad",
                                         "TreeRNN_Regularization_L201",
                                         "TreeRNN_Regularization_L2001",
                                         "TreeRNN_Regularization_L20001",
                                         "TreeRNN_Regularization_Dropout10",
                                         "TreeRNN_Regularization_Dropout25",
                                         "TreeRNN_Regularization_Dropout50",
                                         "TreeRNN_Regularization_L200005_Dropout5",
                                         "TreeRNN_Regularization_L20001_Dropout10"])

    deeprnn = get_Exp4_info(models=["DeepRNN"] * 9,
                            dropouts=["0%", "0%", "0%", "0%", "10%", "25%", "50%", "5%", "10%"],
                            l2s=[0, 0.01, 0.001, 0.0001, 0, 0, 0, 0.00005, 0.0001],
                            model_names=["DeepRNN_batch4_decay1_rep100_lr0001_normal_train_conv100_adam",
                                         "DeepRNN_Regularization_L201",
                                         "DeepRNN_Regularization_L2001",
                                         "DeepRNN_Regularization_L20001",
                                         "DeepRNN_Regularization_Dropout10",
                                         "DeepRNN_Regularization_Dropout25",
                                         "DeepRNN_Regularization_Dropout50",
                                         "DeepRNN_Regularization_L200005_Dropout5",
                                         "DeepRNN_Regularization_L20001_Dropout10"])

    treelstm = get_Exp4_info(models=["TreeLSTM"] * 9,
                             dropouts=["0%", "0%", "0%", "0%", "10%", "25%", "50%", "5%", "10%"],
                             l2s=[0, 0.01, 0.001, 0.0001, 0, 0, 0, 0.00005, 0.0001],
                             model_names=["TreeLSTM_batch4_decay1_rep100_lr01_normal_train_conv100_adagrad",
                                          "TreeLSTM_Regularization_L201",
                                          "TreeLSTM_Regularization_L2001",
                                          "TreeLSTM_Regularization_L20001",
                                          "TreeLSTM_Regularization_Dropout10",
                                          "TreeLSTM_Regularization_Dropout25",
                                          "TreeLSTM_Regularization_Dropout50",
                                          "TreeLSTM_Regularization_L200005_Dropout5",
                                          "TreeLSTM_Regularization_L20001_Dropout10"])

    df = pd.DataFrame(treernn)
    df = df.append(pd.DataFrame(deeprnn))
    df = df.append(pd.DataFrame(treelstm))

    print(df.to_latex(
        columns=["Model", "Dropout Rate", "L2 Scalar", "Train Acc", "Val Acc"],
        index=False))


def get_Exp5_info(models, loss_types, dropouts, l2s, model_names):
    acc_train = []
    acc_val = []
    acc_test = []
    models_list = []
    loss_type_list = []
    dropout_list = []
    l2_list = []
    for model_name, loss_type, dropout, l2, model in zip(model_names, loss_types, dropouts, l2s, models):
        if os.path.exists(f"../trained_models/{model_name}"):
            if os.path.exists(f"../trained_models/{model_name}/performance_train.csv"):
                train = helper.load_dict(f"../trained_models/{model_name}/performance_train.csv")
                val = helper.load_dict(f"../trained_models/{model_name}/performance_val.csv")
                test = helper.load_dict(f"../trained_models/{model_name}/performance_test.csv")
                acc_train.append(round(train["accuracy"], 4))
                acc_val.append(round(val["accuracy"], 4))
                acc_test.append(round(test["accuracy"], 4))
                loss_type_list.append(loss_type)
                dropout_list.append(dropout)
                l2_list.append(l2)
                models_list.append(model)
            else:
                print(model_name, "does not have performance file")
        else:
            print(model_name, "does not exists")

    return {
        "Model": models_list,
        "Loss type": loss_type_list,
        "Dropout Rate": dropout_list,
        "L2 Scalar": l2_list,
        "Train Acc": acc_train,
        "Val Acc": acc_val,
        "Test Acc": acc_test,
    }


def get_Exp5():
    treernn = get_Exp5_info(models=["TreeRNN"] * 3,
                            loss_types=["Internal nodes", "Root node", "Root node"],
                            dropouts=["0%", "0%", "0%"],
                            l2s=[0.001, 0.001, 0],
                            model_names=["TreeRNN_Regularization_L2001",
                                         "TreeRNN_RootLoss",
                                         "TreeRNN_RootLoss_reg0"])

    deeprnn = get_Exp5_info(models=["DeepRNN"] * 3,
                            loss_types=["Internal nodes", "Root node", "Root node"],
                            dropouts=["10%", "10%", "0%"],
                            l2s=[0, 0, 0],
                            model_names=["DeepRNN_Regularization_Dropout10",
                                         "DeepRNN_RootLoss",
                                         "DeepRNN_RootLoss_drop0"])

    treelstm = get_Exp5_info(models=["TreeLSTM"] * 3,
                             loss_types=["Internal nodes", "Root node", "Root node"],
                             dropouts=["10%", "10%", "0%"],
                             l2s=[0, 0, 0],
                             model_names=["TreeLSTM_Regularization_Dropout10",
                                          "TreeLSTM_RootLoss",
                                          "TreeLSTM_RootLoss_drop0"])

    df = pd.DataFrame(treernn)
    df = df.append(pd.DataFrame(deeprnn))
    df = df.append(pd.DataFrame(treelstm))

    print(df.to_latex(
        columns=["Model", "Loss type", "Dropout Rate", "L2 Scalar", "Train Acc", "Val Acc"],
        index=False))


def get_Exp6_info(models, word_methods, min_counts, model_names, model_names2):
    acc_train = []
    acc_val = []
    acc_val2 = []
    val_acc_avg_list = []
    acc_test = []
    models_list = []
    word_method_list = []
    min_count_list = []
    for model_name, model_name2, word_method, min_count, model in zip(model_names, model_names2, word_methods,
                                                                      min_counts, models):
        if os.path.exists(f"../trained_models/{model_name}"):
            if os.path.exists(f"../trained_models/{model_name}/performance_train.csv"):
                train = helper.load_dict(f"../trained_models/{model_name}/performance_train.csv")
                val = helper.load_dict(f"../trained_models/{model_name}/performance_val.csv")
                test = helper.load_dict(f"../trained_models/{model_name}/performance_test.csv")

                val_acc_1 = val["accuracy"]
                val_acc_2 = 0
                if os.path.exists(f"../trained_models/{model_name2}"):
                    if os.path.exists(f"../trained_models/{model_name2}/performance_train.csv"):
                        val2 = helper.load_dict(f"../trained_models/{model_name2}/performance_val.csv")
                        val_acc_2 = val2["accuracy"]
                    else:
                        print(model_name2, "does not have performance file")
                else:
                    print(model_name2, "does not exists")
                acc_val2.append(round(val_acc_2, 4))
                val_acc_avg_list.append(round((val_acc_1 + val_acc_2) / 2, 4))
                acc_train.append(round(train["accuracy"], 4))
                acc_val.append(round(val_acc_1, 4))
                acc_test.append(round(test["accuracy"], 4))
                word_method_list.append(word_method)
                min_count_list.append(min_count)
                models_list.append(model)
            else:
                print(model_name, "does not have performance file")
        else:
            print(model_name, "does not exists")

    return {
        "Model": models_list,
        "Word Embedding Method": word_method_list,
        "Min count": min_count_list,
        "Train Acc": acc_train,
        "Val Acc Run 1": acc_val,
        "Val Acc Run 2": acc_val2,
        "Val Acc Avg": val_acc_avg_list,
        "Test Acc": acc_test,
    }


def get_Exp6():
    treernn = get_Exp6_info(models=["TreeRNN"] * 7,
                            word_methods=["GloVe Pretrained", "GloVe Pretrained", "GloVe Pretrained 840B",
                                          "Word2Vec Pretrained", "fastText Pretrained", "GloVe Trained",
                                          "GloVe Finetuned"],
                            min_counts=[100, 50, 50,
                                        50, 50, 100,
                                        50],
                            model_names=["?",
                                         "TreeRNN_WordEmbed_GloVe_Pretrained_MinCount50",
                                         "TreeRNN_WordEmbed_GloVe_Pretrained_840B",
                                         "TreeRNN_WordEmbed_Word2Vec_Pretrained",
                                         "TreeRNN_WordEmbed_fastText_Pretrained",
                                         "TreeRNN_WordEmbed_GloVe_Trained",
                                         "TreeRNN_WordEmbed_GloVe_Finetuned"],
                            model_names2=["?_V2",
                                          "TreeRNN_WordEmbed_GloVe_Pretrained_MinCount50_V2",
                                          "TreeRNN_WordEmbed_GloVe_Pretrained_840B_V2",
                                          "TreeRNN_WordEmbed_Word2Vec_Pretrained_V2",
                                          "TreeRNN_WordEmbed_fastText_Pretrained_V2",
                                          "TreeRNN_WordEmbed_GloVe_Trained_V2",
                                          "TreeRNN_WordEmbed_GloVe_Finetuned_V2"])

    deeprnn = get_Exp6_info(models=["DeepRNN"] * 7,
                            word_methods=["GloVe Pretrained", "GloVe Pretrained", "GloVe Pretrained 840B",
                                          "Word2Vec Pretrained", "fastText Pretrained", "GloVe Trained",
                                          "GloVe Finetuned"],
                            min_counts=[100, 50, 50,
                                        50, 50, 100,
                                        50],
                            model_names=["?",
                                         "DeepRNN_WordEmbed_GloVe_Pretrained_MinCount50",
                                         "DeepRNN_WordEmbed_GloVe_Pretrained_840B_V1",
                                         "DeepRNN_WordEmbed_Word2Vec_Pretrained",
                                         "DeepRNN_WordEmbed_fastText_Pretrained",
                                         "DeepRNN_WordEmbed_GloVe_Trained",
                                         "DeepRNN_WordEmbed_GloVe_Finetuned"],
                            model_names2=["?_V2",
                                          "DeepRNN_WordEmbed_GloVe_Pretrained_MinCount50_V2",
                                          "DeepRNN_WordEmbed_GloVe_Pretrained_840B_V2",
                                          "DeepRNN_WordEmbed_Word2Vec_Pretrained_V2",
                                          "DeepRNN_WordEmbed_fastText_Pretrained_V2",
                                          "DeepRNN_WordEmbed_GloVe_Trained_V2",
                                          "DeepRNN_WordEmbed_GloVe_Finetuned_V2"])

    treelstm = get_Exp6_info(models=["TreeLSTM"] * 7,
                             word_methods=["GloVe Pretrained", "GloVe Pretrained", "GloVe Pretrained 840B",
                                           "Word2Vec Pretrained", "fastText Pretrained", "GloVe Trained",
                                           "GloVe Finetuned"],
                             min_counts=[100, 50, 50,
                                         50, 50, 100,
                                         50],
                             model_names=["?",
                                          "TreeLSTM_WordEmbed_GloVe_Pretrained_MinCount50_V1",
                                          "TreeLSTM_WordEmbed_GloVe_Pretrained_840B",
                                          "TreeLSTM_WordEmbed_Word2Vec_Pretrained",
                                          "TreeLSTM_WordEmbed_fastText_Pretrained",
                                          "TreeLSTM_WordEmbed_GloVe_Trained",
                                          "TreeLSTM_WordEmbed_GloVe_Finetuned"],
                             model_names2=["?_V2",
                                           "TreeLSTM_WordEmbed_GloVe_Pretrained_MinCount50_V2",
                                           "TreeLSTM_WordEmbed_GloVe_Pretrained_840B_V2",
                                           "TreeLSTM_WordEmbed_Word2Vec_Pretrained_V2",
                                           "TreeLSTM_WordEmbed_fastText_Pretrained_V2",
                                           "TreeLSTM_WordEmbed_GloVe_Trained_V2",
                                           "TreeLSTM_WordEmbed_GloVe_Finetuned_V2"])

    df = pd.DataFrame(treernn)
    df = df.append(pd.DataFrame(deeprnn))
    df = df.append(pd.DataFrame(treelstm))

    print(df.to_latex(
        columns=["Model", "Word Embedding Method", "Min count", "Val Acc Run 1", "Val Acc Run 2", "Val Acc Avg"],
        index=False))


# get_Exp6()

def get_exp_speed_info(model, model_names):
    best_time_list = []
    best_epoch_list = []
    total_time_list = []
    total_epoch_list = []
    for model_name in model_names:
        if os.path.exists(f"../trained_models/{model_name}"):
            if os.path.exists(f"../trained_models/{model_name}/performance_train.csv"):
                speed = helper.load_dict(f"../trained_models/{model_name}/speed.csv")
                best_time_list.append(speed["best_time"] / 60 / 60)
                best_epoch_list.append(speed["best_epoch"])
                total_time_list.append(speed["total_time"] / 60 / 60)
                total_epoch_list.append(speed["epoch"])
            else:
                print(model_name, "does not have performance file")
        else:
            print(model_name, "does not exists")

    return {
        "Model": [model],
        "Avg best epochs": [np.average(best_epoch_list)],
        "Max best epochs": [np.max(best_epoch_list)],
        "Min best epochs": [np.min(best_epoch_list)],
        "Avg total epochs": [np.average(total_epoch_list)],
        "Max total epochs": [np.max(total_epoch_list)],
        "Min total epochs": [np.min(total_epoch_list)],
        "Avg best time": [np.average(best_time_list)],
        "Max best time": [np.max(best_time_list)],
        "Min best time": [np.min(best_time_list)],
        "Avg total time": [np.average(total_time_list)],
        "Max total time": [np.max(total_time_list)],
        "Min total time": [np.min(total_time_list)]
    }


def get_exp_speed():
    treernn = get_exp_speed_info(model="TreeRNN",
                                 model_names=["TreeRNN_batch4_decay98_rep100_lr1_normal_train_conv100_adagrad",
                                              "TreeRNN_batch16_decay98_rep100_lr1_normal_train_conv100_adagrad",
                                              "TreeRNN_batch64_decay98_rep100_lr1_normal_train_conv100_adagrad",

                                              "TreeRNN_batch4_decay98_rep100_lr1_normal_train_conv100_adagrad",
                                              "TreeRNN_batch4_decay995_rep100_lr1_normal_train_conv100_adagrad",
                                              "TreeRNN_batch4_decay1_rep100_lr1_normal_train_conv100_adagrad",
                                              "TreeRNN_batch4_decay98_rep100_lr01_normal_train_conv100_adagrad",
                                              "TreeRNN_batch4_decay995_rep100_lr01_normal_train_conv100_adagrad",
                                              "TreeRNN_batch4_decay1_rep100_lr01_normal_train_conv100_adagrad",

                                              "TreeRNN_batch4_decay995_rep100_lr01_normal_train_conv100_adagrad",
                                              "TreeRNN_batch4_decay1_rep100_lr001_normal_train_conv100_adam",
                                              "TreeRNN_batch4_decay1_rep100_lr0001_normal_train_conv100_adam"
                                              ])

    mtreernn = get_exp_speed_info(model="MTreeRNN",
                                  model_names=["MTreeRNN_batch16_decay98_rep100_lr1_normal_train_conv100_adagrad",
                                               "MTreeRNN_batch16_decay995_rep100_lr1_normal_train_conv100_adagrad",
                                               "MTreeRNN_batch16_decay1_rep100_lr1_normal_train_conv100_adagrad",
                                               "MTreeRNN_batch16_decay98_rep100_lr01_normal_train_conv100_adagrad",
                                               "MTreeRNN_batch16_decay995_rep100_lr01_normal_train_conv100_adagrad",
                                               "MTreeRNN_batch16_decay1_rep100_lr01_normal_train_conv100_adagrad"])

    deepRNN = get_exp_speed_info(model="DeepRNN",
                                 model_names=["DeepRNN_batch4_decay98_rep100_lr1_normal_train_conv100_adagrad",
                                              "DeepRNN_batch4_decay995_rep100_lr1_normal_train_conv100_adagrad",
                                              "DeepRNN_batch4_decay1_rep100_lr1_normal_train_conv100_adagrad",
                                              "DeepRNN_batch4_decay98_rep100_lr01_normal_train_conv100_adagrad",
                                              "DeepRNN_batch4_decay995_rep100_lr01_normal_train_conv100_adagrad",
                                              "DeepRNN_batch4_decay1_rep100_lr01_normal_train_conv100_adagrad"])

    treeLSTM = get_exp_speed_info(model="TreeLSTM",
                                  model_names=["TreeLSTM_batch4_decay98_rep100_lr1_normal_train_conv100_adagrad",
                                               "TreeLSTM_batch4_decay995_rep100_lr1_normal_train_conv100_adagrad",
                                               "TreeLSTM_batch4_decay1_rep100_lr1_normal_train_conv100_adagrad",
                                               "TreeLSTM_batch4_decay98_rep100_lr01_normal_train_conv100_adagrad",
                                               "TreeLSTM_batch4_decay995_rep100_lr01_normal_train_conv100_adagrad",
                                               "TreeLSTM_batch4_decay1_rep100_lr01_normal_train_conv100_adagrad"])

    treeLSTM_tacker = get_exp_speed_info(model="TreeLSTM with Tracker",
                                         model_names=["Tracker_batch16_decay98_rep100_lr1_normal_train_conv100_adagrad",
                                                      "Tracker_batch16_decay995_rep100_lr1_normal_train_conv100_adagrad",
                                                      "Tracker_batch16_decay1_rep100_lr1_normal_train_conv100_adagrad",
                                                      "Tracker_batch16_decay98_rep100_lr01_normal_train_conv100_adagrad",
                                                      "Tracker_batch16_decay995_rep100_lr01_normal_train_conv100_adagrad",
                                                      "Tracker_batch16_decay1_rep100_lr01_normal_train_conv100_adagrad"])

    LSTM = get_exp_speed_info(model="LSTM",
                              model_names=["LSTM_batch4_decay98_rep100_lr1_normal_train_conv100_adagrad",
                                           "LSTM_batch4_decay995_rep100_lr1_normal_train_conv100_adagrad",
                                           "LSTM_batch4_decay1_rep100_lr1_normal_train_conv100_adagrad",
                                           "LSTM_batch4_decay98_rep100_lr01_normal_train_conv100_adagrad",
                                           "LSTM_batch4_decay995_rep100_lr01_normal_train_conv100_adagrad",
                                           "LSTM_batch4_decay1_rep100_lr01_normal_train_conv100_adagrad"])

    df = pd.DataFrame(treernn)
    df = df.append(pd.DataFrame(mtreernn))
    df = df.append(pd.DataFrame(deepRNN))
    df = df.append(pd.DataFrame(treeLSTM))
    df = df.append(pd.DataFrame(treeLSTM_tacker))
    df = df.append(pd.DataFrame(LSTM))

    print(df.to_latex(columns=["Model",
                               "Avg best epochs",
                               "Max best epochs",
                               "Min best epochs",
                               "Avg total epochs",
                               "Max total epochs",
                               "Min total epochs",
                               "Avg best time",
                               "Max best time",
                               "Min best time",
                               "Avg total time",
                               "Max total time",
                               "Min total time"],
                      index=False))


get_exp_speed()
