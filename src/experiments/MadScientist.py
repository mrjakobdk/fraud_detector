from utils import data_util


class Experiment():
    def __init__(self, model, data, word_embed, lr, lr_decay, lr_end, conv_cond):
        self.model = model
        self.data = data
        self.word_embed = word_embed
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_end = lr_end
        self.conv_cond = conv_cond

    def run(self):
        model = self.model(data, word_embed, lr, lr_decay, )



class MadScientist():
    def __init__(self):
        pass

    def run_experiments(self, list_of_experiments):
        for experiment in list_of_experiments:
            experiment.run()
