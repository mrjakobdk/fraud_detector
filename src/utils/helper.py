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

def reverse_dict(l):
    n = len(l)
    rev_l = dict()
    for i in range(n):
        rev_l[l[i]] = i
    return rev_l
