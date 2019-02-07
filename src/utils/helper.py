from utils.flags import FLAGS


def _print(*args):
    if FLAGS.verbose:
        print(args)