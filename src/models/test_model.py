import torch
from alpha_zero_model import Net as alphaNet
import alpha_zero_model as azm

def testAlphaZero():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import random
    state1 = [[random.choice([0, 1]) for _ in range(61)] for _ in range(41)]
    state2 = [[random.choice([0, 1]) for _ in range(61)] for _ in range(41)]

    state = [[state1], [state2]]

    net = alphaNet(azm.STATE_SHAPE, 20, 12, 'cuda').to(device)
    tensor = net.state_lists_to_batch(state)
    print(tensor.shape)
    pol, val = net(tensor)
    print(pol)
    print(val)

def testRL():
    pass

if __name__ == "__main__":
    testAlphaZero()