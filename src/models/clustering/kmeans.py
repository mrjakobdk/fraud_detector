from time import time
from sklearn.cluster import KMeans as KM
from models.clustering.clusterModel import ClusterModel
from utils import helper


class KMeans(ClusterModel):

    def cluster(self, inputs):
        t = time()
        helper._print('Training clusters (KMeans)...')
        kmeans = KM(n_clusters=self.num_clusters, init=self.cluster_init, max_iter=1000, tol=0.000001)
        cluster_pred = kmeans.fit_predict(inputs)
        helper._print(f'Done training clusters. Finished in {int((time() - t)/60)} minutes and {int((time() - t) % 60)} seconds!')

        return cluster_pred
