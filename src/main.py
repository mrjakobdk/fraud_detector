import tensorflow as tf
import utils.data_util as data_util
from models.treeRNN import treeRNN
from models.treeRNN_batch import treeRNN_GPU
from models.deepRNN import deepRNN
import trainers.TreeTrainer as trainer
from utils.flags import FLAGS
from experiments import SpeedTester


def main():
    if FLAGS.run_speed_test:
        batch_sizes = [2 ** i for i in range(1, 10)]
        # SpeedTester.run(
        #     [treeRNN, treeRNN_GPU],
        #     batch_sizes,
        #     [tf.ConfigProto(device_count={'GPU': 0}), None]
        # )
        SpeedTester.plot(batch_sizes, ["Neerbek - CPU", "Neerbek - GPU", "Our - CPU", "Our - GPU"])
        return

    _data_util = data_util.DataUtil()
    data = _data_util.get_data()

    if FLAGS.use_gpu:
        config = None
    else:
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )

    if FLAGS.model == "deepRNN":
        model = deepRNN(data, FLAGS.models_dir + FLAGS.model_name + "model.ckpt")
    elif FLAGS.model == "treeRNN_batch":
        model = treeRNN_GPU(data, FLAGS.models_dir + FLAGS.model_name + "model.ckpt")
    else:
        model = treeRNN(data, FLAGS.models_dir + FLAGS.model_name + "model.ckpt")

    trainer.train(model, load=False, config=config)


if __name__ == "__main__":
    main()
