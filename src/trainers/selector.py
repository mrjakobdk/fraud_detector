from time import time

from tqdm import tqdm

from models.clustering.Agglomerative import Agglomerative
from models.clustering.DBSCAN import DBSCAN
from models.clustering.kmeans import KMeans
from utils import helper, performance
from utils.flags import FLAGS
import numpy as np


class Selector:
    def __init__(self, model, session, num_clusters=FLAGS.num_clusters, cluster_model=FLAGS.cluster_model):

        self.num_clusters = num_clusters
        self.model = model
        self.session = session
        if cluster_model == 'dbscan':
            self.cluster_model = DBSCAN(self.num_clusters)
        elif cluster_model == 'agglo':
            self.cluster_model = Agglomerative(self.num_clusters)
        else:
            self.cluster_model = KMeans(self.num_clusters)

    def select_data(self, data, cut_off):
        # Get representations
        t = time()
        representations, predictions, labels, permutations = [], [], [], []
        batch_size = 3000
        batches = helper.batches(data, batch_size, perm=False)
        pbar = tqdm(
            bar_format='{percentage:.0f}%|{bar}| Elapsed: {elapsed}, Remaining: {remaining} (batches: {n_fmt}/{total_fmt}) ',
            total=len(batches))
        for i, batch in enumerate(batches):
            feed_dict, permuts = self.model.build_feed_dict(batch)
            reps, preds, labs = self.session.run(
                [self.model.sentence_representations, self.model.p, self.model.labels], feed_dict=feed_dict)
            representations.extend(reps)
            predictions.extend(preds)
            labels.extend(labs)
            permutations.extend(list(i * batch_size + np.array(permuts)))
            pbar.update(1)
        pbar.close()
        print()

        self.representations = np.array(representations)[permutations]
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
        return data_to_use

    def mfo(self, cluster):
        cluster_labels = self.labels[self.cluster_predictions == cluster]
        bincount = np.bincount(cluster_labels)
        if len(cluster_labels) == 0:
            return 0
        return bincount.max() / len(cluster_labels)