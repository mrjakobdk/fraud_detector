from time import sleep
from tqdm import tqdm
from utils import directories

with open(directories.ENRON_TRAIN_SENTENCES_TXT_PATH, 'r', encoding='utf-8') as train_sentences:
    count = 0
    lines = train_sentences.readlines()
    pbar = tqdm(
        bar_format='{percentage:.0f}%|{bar}| Elapsed: {elapsed}, Remaining: {remaining} ({n_fmt}/{total_fmt}) ',
        total=len(lines))
    for index, line in enumerate(lines):
        count += len(line.split())
        pbar.update(1)
    pbar.close()
    print('Final count: %i' %count)