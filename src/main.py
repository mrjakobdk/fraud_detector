import tensorflow as tf
import utils.data_util as data_util
from models.trees.treeLSTM import treeLSTM
from models.trees.treeRNN import treeRNN
from models.trees.treeRNN_batch import treeRNN_batch
from models.trees.deepRNN import deepRNN
import trainers.TreeTrainer as trainer
from models.trees.treeRNN_neerbek import treeRNN_neerbek
from utils.flags import FLAGS
from experiments import SpeedTester


def main():

    _data_util = data_util.DataUtil()
    data = _data_util.get_data()

    if FLAGS.use_gpu:
        config = None
    else:
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )

    if FLAGS.model_name == "":
        FLAGS.model_name = FLAGS.model + \
                           "_BathcSize" + str(FLAGS.batch_size) + \
                           "_LrStart" + str(FLAGS.learning_rate) + \
                           "_LrEnd" + str(FLAGS.learning_rate_end) + \
                           "_ExpDecay" + str(FLAGS.lr_decay) + \
                           "_ConvCond" + str(FLAGS.conv_cond) + \
                           "/"

    model_placement = FLAGS.models_dir + FLAGS.model_name + "model.ckpt"
    if FLAGS.model == "deepRNN":
        model = deepRNN(data, model_placement)
    elif FLAGS.model == "treeRNN_batch":
        model = treeRNN_batch(data, model_placement)
    elif FLAGS.model == "treeRNN_neerbek":
        model = treeRNN_neerbek(data, model_placement)
    elif FLAGS.model == "treeLSTM":
        model = treeLSTM(data, model_placement)
    elif FLAGS.model == "treeRNN_tracker":
        model = treeRNN_tracker(data, model_placement)
    else:
        model = treeRNN(data, model_placement)

    trainer.train(model, load=False, config=config)


if __name__ == "__main__":
    main()
