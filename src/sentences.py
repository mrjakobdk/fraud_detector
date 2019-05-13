from utils import data_util, tree_util, helper

_data_util = data_util.DataUtil()
data = _data_util.get_data()
roots_size = [tree_util.size_of_tree(root) for root in data.train_trees]
roots = helper.sort_by(data.train_trees, roots_size)

for root in roots[-5:]:
    print(root.label)
    print(root.to_sentence())
    print()