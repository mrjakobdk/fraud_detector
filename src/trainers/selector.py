from time import time
from models.clustering.kmeans import KMeans
from utils import helper, performance
from utils.flags import FLAGS
import numpy as np


class Selector:
    def __init__(self, model, session, num_clusters=FLAGS.num_clusters, cluster_model=FLAGS.cluster_model):

        self.num_clusters = num_clusters
        self.model = model
        self.session = session
        if cluster_model == 'kmeans':
            self.cluster_model = KMeans(self.num_clusters)

    def select_data(self, data, cut_off):
        # Get representations
        t = time()
        representations, predictions, labels, permutations = [], [], [], []
        batch_size = 3000
        for i, batch in enumerate(helper.batches(data, batch_size, perm=False)):
            feed_dict, permuts = self.model.build_feed_dict(batch)
            reps, preds, labs = self.session.run(
                [self.model.sentence_representations, self.model.p, self.model.labels], feed_dict=feed_dict)
            representations.extend(reps)
            predictions.extend(preds)
            labels.extend(labs)
            permutations.extend(list(i * batch_size + np.array(permuts)))

        self.representations = np.array(representations)[permutations]
        print(representations)
        self.predictions = performance.get_prediction(np.array(predictions)[permutations])
        self.labels = performance.get_prediction(np.array(labels)[permutations])
        # Get clusters

        self.cluster_predictions = self.cluster_model.cluster(self.representations)

        # Get acc of clusters
        cluster_acc = []
        for i in range(self.num_clusters):
            if FLAGS.mfo:
                acc = self.mfo(i)
            else:
                acc = performance.get_accuracy(self.labels[self.cluster_predictions == i], self.predictions[self.cluster_predictions == i])
            cluster_acc.append((i, acc))

        # Return data
        cluster_acc.sort(key=lambda el: el[1], reverse=True)
        helper._print(f'Cluster accuracies(MFO={FLAGS.mfo}):')
        for k, acc in cluster_acc:
            helper._print(f'\tCluster {k}: {acc}, size: {len(self.labels[self.cluster_predictions == k])}/{len(data)}')

        removed_percent = 0
        data_to_use = []
        for cluster, acc in cluster_acc:
            new_percent = removed_percent + len(self.predictions[self.cluster_predictions == cluster]) / len(self.predictions)
            if removed_percent > cut_off or (new_percent > cut_off and abs(new_percent - cut_off) > abs(removed_percent - cut_off)):
                data_to_use.extend(data[self.cluster_predictions == cluster])
            removed_percent = new_percent

        helper._print(
            f'Done selecting data for training. Overall time used for selection is {int((time() - t)/60)} minutes and {int((time() - t) % 60)} seconds')
        helper._print(
            f'Using {len(data_to_use)}/{len(data)} ({len(data_to_use)/len(data)*100}%) for the next {FLAGS.select_freq} epochs')
        return data_to_use

    def mfo(self, cluster):
        cluster_labels = self.labels[self.cluster_predictions == cluster]
        bincount = np.bincount(cluster_labels)
        return bincount.max() / len(cluster_labels)