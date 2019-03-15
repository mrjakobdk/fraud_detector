from experiments import SpeedTester
from utils.flags import FLAGS

if FLAGS.run_speed_test:
    batch_sizes = [2 ** i for i in range(1, 10)]
    # SpeedTester.run1(
    #     [treeRNN, treeRNN_GPU],
    #     batch_sizes,
    #     [tf.ConfigProto(device_count={'GPU': 0}), None]
    # )
    SpeedTester.plot1(batch_sizes, ["Neerbek - CPU", "Neerbek - GPU", "Our - CPU", "Our - GPU"])


SpeedTester.run2()