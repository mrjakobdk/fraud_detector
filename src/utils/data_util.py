
from utils.data import Data

class DataUtil:
    def __init__(self):
        self.data = Data()

    def get_data(self):
        data = Data()
        return data

    def parse_sentence(self, sentence):
        return self.nlp_client.annotate(sentence)

    def generate_input_arrays(self):
        for d in self.data:
            parseTree = self.parse_sentence(d).sentence[0].parseTree
            # print(parseTree)


