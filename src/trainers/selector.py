from time import time
from models.clustering.kmeans import KMeans
from utils import helper, performance
from utils.flags import FLAGS


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
        feed_dict, permutations = self.model.build_feed_dict(data)
        representations, predictions, labels = self.session.run([self.model.sentence_representations, self.model.p, self.model.labels], feed_dict=feed_dict)

        representations = representations[permutations]
        predictions = predictions[permutations]
        labels = labels[permutations]
        # Get clusters
        cluster_predictions = self.cluster_model.cluster(representations)


        # Get acc of clusters
        cluster_acc = []
        for i in range(self.num_clusters):
            acc = performance.get_accuracy(labels[cluster_predictions == i], predictions[cluster_predictions == i])
            cluster_acc.append((i, acc))


        # Return data
        print(cluster_acc)
        cluster_acc.sort(key=lambda el:  el[1], reverse=True)
        helper._print('Cluster accuracies:')
        for k, acc in cluster_acc:
            helper._print(f'\tCluster {k}: {acc}')
        print(cluster_acc)

        removed_percent = 0
        data_to_use = []
        for cluster, acc in cluster_acc:
            removed_percent += len(predictions[cluster_predictions == cluster])/len(predictions)
            if removed_percent > cut_off:
                data_to_use += data[cluster_predictions == cluster]

        helper._print(f'Done selecting data for training. Overall time used for selection is {int((time() - t)/60)} minutes and {(time() - t) % 60} seconds')
        return data_to_use


