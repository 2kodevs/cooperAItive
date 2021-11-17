# Load a light checkpoint and build a model with the actual
# implementation of AlphaZeroNet
#
#   arg 1:
#       Path to the checkpoints folder
#   arg 2:
#       Tag of the checkpoint
#   arg 2:
#       Save tag of the builded model

from ..players import AlphaZeroNet
import sys

if __name__ == '__main__':
    path      = sys.argv[1]
    load_tag  = sys.argv[2]
    save_tag  = sys.argv[3]

    net = AlphaZeroNet()
    config = net.load_checkpoint(path, load_tag)
    net.save(None, config, None, path, True, tag=save_tag, verbose=True)
