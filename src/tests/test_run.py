from utils import data_util
from models.trees.treeRNN_batch import treeRNN
from utils.flags import FLAGS
import trainers.TreeTrainer as trainer

_data_util = data_util.DataUtil()
data = _data_util.get_data()

model = treeRNN(data, FLAGS.models_dir + FLAGS.model_name + "model.ckpt")
trainer.train(model, load=False)