import utils.data_util as data_util
import models.treeRNN as tRNN
# import trained_models.treeRNN_batch_friendly as tRNN_gpu
from utils.flags import FLAGS

def main():
    _data_util = data_util.DataUtil()
    data = _data_util.get_data()

    # if FLAGS.use_gpu:
    #     _tRNN = tRNN_gpu.tRNN(data)
    # else:
    _tRNN = tRNN.tRNN(data)
    _tRNN.train()

if __name__ == "__main__":
    main()