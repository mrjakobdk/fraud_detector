import utils.data_util as data_util
import models.treeRNN as tRNN

def main():
    _data_util = data_util.DataUtil()
    data = _data_util.get_data()

    _tRNN = tRNN.tRNN(data)
    _tRNN.train()

if __name__ == "__main__":
    main()