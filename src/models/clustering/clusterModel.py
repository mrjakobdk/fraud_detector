from utils.flags import FLAGS


class ClusterModel:
    def __init__(self, num_clusters, cluster_init=FLAGS.cluster_initialization):
        self.num_clusters = num_clusters
        self.cluster_init = cluster_init


    def cluster(self, inputs):
        # Trains new clusters on the input and returns array of which clusters each input belongs to
        raise NotImplementedError("Each Model must re-implement this method.")


