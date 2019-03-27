import numpy as np

from time import time
from sklearn.cluster import DBSCAN as DB
from models.clustering.clusterModel import ClusterModel
from utils import helper


class DBSCAN(ClusterModel):

    def cluster(self, inputs):
        t = time()
        helper._print('Training clusters (DBSCAN)...')

        epsilon = 0.0005
        prev_epsilon = epsilon

        cluster_pred = []

        min_clusters = self.num_clusters - 2
        max_clusters = self.num_clusters + 2
        num_clusters = 0

        helper._print('Exponentially increasing the epsilon...')

        while min_clusters > num_clusters:
            prev_epsilon = epsilon
            epsilon *= 2
            dbscan = DB(eps=epsilon, min_samples=5)
            cluster_pred = dbscan.fit_predict(inputs)
            num_clusters = len(np.unique(cluster_pred))
            helper._print(f'Number of clusters: {num_clusters}, Epsilon: {epsilon}')

        high_epsilon = epsilon
        low_epsilon = prev_epsilon

        helper._print('Using binary search to find the right epsilon...')

        while max_clusters < num_clusters or num_clusters < min_clusters:
            new_epsilon = (low_epsilon + high_epsilon)/2
            dbscan = DB(eps=new_epsilon, min_samples=5)
            cluster_pred = dbscan.fit_predict(inputs)
            num_clusters = len(np.unique(cluster_pred))
            if num_clusters < min_clusters:
                high_epsilon = new_epsilon
            if num_clusters > max_clusters:
                low_epsilon = new_epsilon
            helper._print(f'Number of clusters: {num_clusters}, Epsilon: {low_epsilon} (low) / {high_epsilon} (high)')

        helper._print(f'Done training clusters. Finished in {int((time() - t)/60)} minutes and {int((time() - t) % 60)} seconds!')

        return cluster_pred
