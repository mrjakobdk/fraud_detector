from utils.flags import FLAGS
from tqdm import tqdm
import urllib


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def _print(*args):
    if FLAGS.verbose:
        print(*args)


def _print_header(text, total=80):
    n = len(text)
    padding_size = int((total - n) / 2) - 1
    padding_left = "=" * padding_size
    padding_right = "=" * (padding_size + (1 if (n - total) % 2 == 1 else 0))
    print(padding_left, text, padding_right)


def reverse_dict(l):
    n = len(l)
    rev_l = dict()
    for i in range(n):
        rev_l[l[i]] = i
    return rev_l


def batches(data_list, batch_size):
    batch_list = []
    batch = []
    for step, data in enumerate(data_list):
        batch.append(data)
        if (step + 1) % batch_size == 0:
            batch_list.append(batch)
            batch = []
    if (step + 1) % batch_size != 0:
        batch_list.append(batch)
    return batch_list

def flatten(l):
    return [item for sublist in l for item in sublist]