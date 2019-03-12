from utils.flags import FLAGS
from tqdm import tqdm
import urllib
from kaggle.api.kaggle_api_extended import KaggleApi
import numpy as np
import sys


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_from_kaggle(data_name, dest):
    api = KaggleApi()
    api.authenticate()
    return api.dataset_download_files(data_name, dest)


def _print(*args):
    if FLAGS.verbose:
        print(*args)


def _print_header(text, total=80):
    n = len(text)
    padding_size = int((total - n) / 2) - 1
    padding_left = "=" * padding_size
    padding_right = "=" * (padding_size + (1 if (n - total) % 2 == 1 else 0))
    print(padding_left, text, padding_right)


def _print_subheader(text, total=80):
    n = len(text)
    padding_size = int((total - n) / 2) - 1
    padding_left = "-" * padding_size
    padding_right = "-" * (padding_size + (1 if (n - total) % 2 == 1 else 0))
    print(padding_left, text, padding_right)


def reverse_dict(l):
    n = len(l)
    rev_l = dict()
    for i in range(n):
        rev_l[l[i]] = i
    return rev_l


def batches(data_list, batch_size, use_tail=True, perm=True):
    if perm:
        data_list = np.random.permutation(data_list)
    batch_list = []
    batch = []
    for step, data in enumerate(data_list):
        batch.append(data)
        if (step + 1) % batch_size == 0:
            batch_list.append(batch)
            batch = []
    if use_tail and ((step + 1) % batch_size != 0):
        batch_list.append(batch)
    return batch_list


def flatten(l):
    return [item for sublist in l for item in sublist]


def add_one(l):
    return [i + 1 for i in l]


def lists_pad(lists, padding):
    max_length = 0
    for l in lists:
        max_length = max(len(l), max_length)

    for i in range(len(lists)):
        lists[i] = lists[i] + [padding] * (max_length - len(lists[i]))

    return lists


def sort_by(X, Y):
    return [x for _, x in sorted(zip(Y, X), key=lambda pair: pair[0])]


def greedy_best_fit(l):
    pass


def greedy_bin_packing(items, values, max):
    bins = [[]]
    bins_size = [0]

    for item, value in zip(items, values):
        found = False
        for i in range(len(bins_size)):
            if bins_size[i] + value <= max:
                bins_size[i] += value
                bins[i].append(item)
                found = True
                break
        if not found:
            bins.append([item])
            bins_size.append(value)

    return bins


def all_combinations(*args):
    return np.array(np.meshgrid(*args)).T.reshape(-1, len(args))