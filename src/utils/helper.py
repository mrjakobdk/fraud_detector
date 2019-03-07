import os
import numpy as np
from utils.flags import FLAGS
from tqdm import tqdm
from functools import wraps
import urllib
from kaggle.api.kaggle_api_extended import KaggleApi
import msgpack
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


def batches(data_list, batch_size, use_tail=False):
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


def listify(fn):
    """
    Use this decorator on a generator function to make it return a list
    instead.
    """

    @wraps(fn)
    def listified(*args, **kwargs):
        return list(fn(*args, **kwargs))

    return listified

def get_or_build(path, build_fn, *args, **kwargs):
    """
    Load from serialized form or build an object, saving the built
    object.
    Remaining arguments are provided to `build_fn`.
    """

    save = False
    obj = None

    if path is not None and os.path.isfile(path):
        _print_subheader('Loading: ' + path)
        with open(path, 'rb') as obj_f:
            obj = msgpack.load(obj_f, use_list=False, encoding='utf-8')
    else:

        save = True

    if obj is None:
        obj = build_fn(*args, **kwargs)

        if save and path is not None:
            with open(path, 'wb') as obj_f:
                msgpack.dump(obj, obj_f)

    return obj
