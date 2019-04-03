
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


def gen_acc_table():

    neerbek = get_models_info(models=["TreeRNN"] * 4,
                              batch_sizes=[4, 4, 64, 64],
                              lr_decay=[0.995, 0.98, 0.995, 0.98],
                              model_names=["treeRNN_neerbek_batch4_decay995_rep100",
                                           "treeRNN_neerbek_batch4_decay98_rep100",
                                           "treeRNN_neerbek_batch64_decay995_rep100",
                                           "treeRNN_neerbek_batch64_decay98_rep100"])

    deepRNN = get_models_info(models=["DeepRNN"] * 4,
                              batch_sizes=[4, 4, 64, 64],
                              lr_decay=[0.995, 0.98, 0.995, 0.98],
                              model_names=["deepRNN_batch4_decay995_rep100",
                                           "deepRNN_batch4_decay98_rep100",
                                           "deepRNN_batch64_decay995_rep100",
                                           "deepRNN_batch64_decay98_rep100"])

    treeLSTM = get_models_info(models=["TreeLSTM"] * 3,
                              batch_sizes=[4, 64, 64],
                              lr_decay=[0.995, 0.995, 0.98],
                              model_names=["treeLSTM_batch4_decay995_rep100",
                                           "treeLSTM_batch64_decay995_rep100",
                                           "treeLSTM_batch64_decay98_rep100"])

    df = pd.DataFrame(neerbek)
    df = df.append(pd.DataFrame(deepRNN))
    df = df.append(pd.DataFrame(treeLSTM))

    print(df.to_latex(columns=["Model", "Batch size", "Lr decay", "acc val", "acc test"], index=False))


gen_acc_table()
