import torch
import torch.nn as nn
import numpy as np

#STATE= [(55bits, 4bits, 2bits) x 41]
STATE_SHAPE = (1, 41, 61)
NUM_FILTERS = 256
KERNEL_SIZE = 3

class Net(nn.Module):
    """
    Neural Network for Alpha Zero implementation of Dominoes
    """
    def __init__(self, input_shape, policy_shap, residual_layers, device='cpu'):
        """
        param input_shape: (int, int, int)
            Dimensions of the input.
        param policy_shape: int
            Number of total actions in policy head
        param residual_layers: int 
            Number of residual convolutionals layers
        param device:
            cpu or cuda
        """
        super(Net, self).__init__()

        device = torch.device('cuda' if device == 'cuda' else 'cpu')
        self.device = device

        self.conv_in = nn.Sequential(
            nn.Conv2d(input_shape[0], NUM_FILTERS, kernel_size=KERNEL_SIZE, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.LeakyReLU()
        ).to(device)

        # layers with residual
        self.residual_nets = []
        for _ in range(residual_layers):
            self.residual_nets.append(nn.Sequential(
                nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=KERNEL_SIZE, padding=1),
                nn.BatchNorm2d(NUM_FILTERS),
                nn.LeakyReLU()
            ).to(device))

        # value head
        self.conv_val = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU()
        ).to(device)

        body_out_shape = (NUM_FILTERS, ) + input_shape[1:]
        conv_val_size = self._get_conv_val_size(body_out_shape)

        self.value = nn.Sequential(
            nn.Linear(conv_val_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        ).to(device)

        # policy head
        self.conv_policy = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, input_shape[0], kernel_size=1),
            nn.BatchNorm2d(input_shape[0]),
            nn.LeakyReLU()
        ).to(device)
        conv_policy_size = self._get_conv_policy_size(body_out_shape)
        self.policy = nn.Sequential(
            nn.Linear(conv_policy_size, policy_shape)
        ).to(device)



    def _get_conv_val_size(self, shape):
        o = self.conv_val(torch.zeros(1, *shape).to(self.device))
        return int(np.prod(o.size()))

    def _get_conv_policy_size(self, shape):
        o = self.conv_policy(torch.zeros(1, *shape).to(self.device))
        return int(np.prod(o.size()))

    def forward(self, x):
        batch_size = x.size()[0]

        v = self.conv_in(x)
        for net in self.residual_nets:
            v = net(v)

        val = self.conv_val(v)
        val = self.value(val.view(batch_size, -1))

        pol = self.conv_policy(v)
        pol = self.policy(pol.view(batch_size, -1))

        return pol, val

    def state_lists_to_batch(self, state_lists):
        """
        Convert list of list states to batch for network

        param state_lists: 
            list of 'list states'

        return 
            States to Tensor
        """
        assert isinstance(state_lists, list)
        batch_size = len(state_lists)
        batch = np.zeros((batch_size,) + STATE_SHAPE, dtype=np.float32)
        for idx, state in enumerate(state_lists):
            batch[idx] = state
        return torch.tensor(batch).to(self.device)
