import tensorflow as tf
import utils.data_util as data_util
from models.treeRNN import treeRNN
from models.treeRNN_batch import treeRNN_batch
from models.deepRNN import deepRNN
import trainers.TreeTrainer as trainer
from utils.flags import FLAGS
from experiments import SpeedTester

SpeedTester.run2()