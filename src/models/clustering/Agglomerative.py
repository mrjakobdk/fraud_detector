from time import time
from sklearn.cluster import AgglomerativeClustering
from models.clustering.clusterModel import ClusterModel
from utils import helper


class Agglomerative(ClusterModel):

    def cluster(self, inputs):
        t = time()
        helper._print('Training clusters (Agglomerative clustering)...')
        agglo = AgglomerativeClustering(n_clusters=self.num_clusters)
        cluster_pred = agglo.fit_predict(inputs)
        helper._print(f'Done training clusters. Finished in {int((time() - t)/60)} minutes and {int((time() - t) % 60)} seconds!')

        return cluster_pred
