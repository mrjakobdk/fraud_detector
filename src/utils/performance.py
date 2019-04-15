import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


def get_labels(data):
    labels = []
    for root in data:
        labels.append(np.argmax(root.label))
    return np.array(labels)


def get_prediction(probs):
    return np.argmax(probs, axis=1)


def get_prediction_at(probs, cutoff):
    return np.array([1 if p[1] > cutoff else 0 for p in probs])


def get_accuracy(labels, predictions):
    return np.mean(labels == predictions)


def get_confusion_matrix(labels, predictions):
    TP = np.sum(predictions[labels == 1])
    FP = np.sum(predictions[labels == 0])
    TN = np.sum(1 - predictions[labels == 0])
    FN = np.sum(1 - predictions[labels == 1])
    return TP, FP, TN, FN


def get_roc_values(labels, probs):
    TPR_list = []
    FPR_list = []
    cutoff = 0
    while cutoff <= 1:
        predictions = get_prediction_at(probs, cutoff)
        TP, FP, TN, FN = get_confusion_matrix(labels, predictions)
        TPR_list.append(TP / (TP + FN))
        FPR_list.append(FP / (TN + FP))
        cutoff += 0.05
    return TPR_list, FPR_list




class Performance:
    def __init__(self, data, model, sess):
        """
        :param data:
        :param model: expect an initialized and trained model
        :return:
        """

        probs, labels = model.predict_and_label(data, sess)
        labels = get_prediction(labels)
        predictions = get_prediction(probs)

        self.acc = get_accuracy(labels, predictions)

        self.TP, self.FP, self.TN, self.FN = get_confusion_matrix(labels, predictions)

        self.precision = self.TP / (self.TP + self.FP)
        self.recall = self.TP / (self.TP + self.FN)
        self.F1 = 2 * (self.precision * self.recall) / (self.precision + self.recall)

        self.TPR_list, self.FPR_list = get_roc_values(labels, probs)
        self.auc = metrics.auc(self.FPR_list, self.TPR_list)


    def get_performance(self):
        performance = {
            "accuracy": self.acc,
            "auc": self.auc,
            "tp": self.TP,
            "fp": self.FP,
            "tn": self.TN,
            "fn": self.FN,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.F1,
        }
        return performance

    def plot_ROC(self, show=False, placement=""):
        plt.clf()
        plt.plot(self.FPR_list, self.TPR_list)
        plt.ylabel("True positive rate")
        plt.xlabel("False positive rate")
        if show:
            plt.show()
        else:
            plt.savefig(placement)
