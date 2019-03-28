import numpy as np
import os

from time import time
from tqdm import tqdm
from models.clustering.Agglomerative import Agglomerative
from models.clustering.DBSCAN import DBSCAN
from models.clustering.kmeans import KMeans
from utils import helper, performance, directories, tree_util
from utils.flags import FLAGS


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

    def select_data(self, data, cut_off, cluster_predictions=None):

        t = time()
        if cluster_predictions is None:

            # Get representations
            representations, predictions, labels, permutations = [], [], [], []
            batch_size = 3000
            batches = helper.batches(data, batch_size, perm=False)
            pbar = tqdm(
                bar_format='{percentage:.0f}%|{bar}| Elapsed: {elapsed}, Remaining: {remaining} (batches: {n_fmt}/{total_fmt}) ',
                total=len(batches))
            for i, batch in enumerate(batches):
                feed_dict, permuts = self.model.build_feed_dict(batch)
                reps, labs = self.session.run(
                    [self.model.sentence_representations, self.model.labels], feed_dict=feed_dict)
                representations.extend(reps)
                labels.extend(labs)
                permutations.extend(list(i * batch_size + np.array(permuts)))
                pbar.update(1)
            pbar.close()
            print()

            self.representations = np.array(representations)[permutations]
            self.labels = performance.get_prediction(np.array(labels)[permutations])

            # Get clusters

            try_cluster = True
            tries = 10
            while try_cluster:
                tries -= 1
                self.cluster_predictions = self.cluster_model.cluster(self.representations)
                if np.bincount(self.cluster_predictions).max() <= 0.8 * len(self.representations) and tries >=0:
                    try_cluster = False

        else:
            self.cluster_predictions = cluster_predictions
            self.labels = tree_util.get_labels(data)

        # Get acc of clusters
        cluster_mfo = []
        for i in range(self.num_clusters):
            mfo = self.mfo(i)
            cluster_mfo.append((i, mfo))

        # Return data
        cluster_mfo.sort(key=lambda el: el[1], reverse=True)
        helper._print(f'Cluster MFO scores:')
        for k, mfo in cluster_mfo:
            helper._print(f'\tCluster {k}: {mfo}, size: {len(self.labels[self.cluster_predictions == k])}/{len(data)}')

        removed_percent = 0
        data_to_use = []
        for cluster, acc in cluster_mfo:
            new_percent = removed_percent + len(data[self.cluster_predictions == cluster]) / len(
                data)
            removed_percent = new_percent
            if acc < cut_off:
                data_to_use.extend(data[self.cluster_predictions == cluster])

        helper._print(
            f'Done selecting data for training. Overall time used for selection is {int((time() - t)/60)} minutes and {int((time() - t) % 60)} seconds')
        return data_to_use, self.cluster_predictions

    def mfo(self, cluster):
        cluster_labels = self.labels[self.cluster_predictions == cluster]
        bincount = np.bincount(cluster_labels)
        if len(cluster_labels) == 0:
            return 0
        return bincount.max() / len(cluster_labels)
