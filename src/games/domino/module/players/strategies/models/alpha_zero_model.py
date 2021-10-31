import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from ..utils import state_to_list

#STATE= [(56bits, 4bits, 2bits) x 41]
STATE_SHAPE = (1, 41, 62)
KERNEL_SIZE = 3

class Net(nn.Module):
    """
    Neural Network for Alpha Zero implementation of Dominoes
    """
    def __init__(self, residual_layers, filters, input_shape=STATE_SHAPE, policy_shape=111, lr=0.001):
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
        self.save_path = 'checkpoints/'
        self.residual_layers = residual_layers
        self.filters = filters

        self.conv_in = nn.Sequential(
            nn.Conv2d(input_shape[0], filters, kernel_size=KERNEL_SIZE, padding=1),
            nn.BatchNorm2d(filters),
            nn.LeakyReLU(),
        )

        # layers with residual
        blocks = []
        for _ in range(residual_layers):
            block = nn.Sequential(
                nn.Conv2d(filters, filters, kernel_size=KERNEL_SIZE, padding=1, bias=False),
                nn.BatchNorm2d(filters),
                nn.LeakyReLU(),
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        # value head
        self.value = nn.Sequential(
            nn.Linear(filters, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

        # policy head
        self.policy = nn.Sequential(
            nn.Linear(filters, policy_shape)
        )

        # colab head
        self.colab = nn.Sequential(
            nn.Linear(filters, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 4),
            nn.Tanh(),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        #optimizer
        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    def forward(self, x):
        v = self.conv_in(x)
        v = F.avg_pool2d(v, v.size(2))

        for b in self.blocks:
            v = b(v)

        v = self.avgpool(v)
        v = v.view(v.size(0), -1)

        pol = self.policy(v)
        val = self.value(v)
        col = self.colab(v)

        return pol, val, col

    def predict(self, s, mask):
        """
        Infer node data given an state

        param s: 
            list of encoded states of the game
        param mask
            list of encoded valids actions from a position of the game
        return
            (Move probabilities P vector, value V vector)
        """
        self.eval()
        batch = self.state_lists_to_batch(s)
        valid_actions_masks = [self.valids_actions_to_tensor(va) for va in mask]
        pol, val, col = self(batch)
        pol = [self.get_policy_value(p, mask, False) for p, mask in zip(pol, valid_actions_masks)]
        return pol, val, col

    def get_policy_value(self, logits, mask, log_softmax):
        """
        Get move probabilities distribution.

        param logits:
            list of 111 bits. Raw policy head
        param mask
            list of 111 bits. Mask of available actions
        param log_softmax
            Set True to use log_softmax as activation function. Set False to use softmax 

        return
            Move probabilities
        """
        selection = torch.masked_select(logits, mask)
        dist = F.log_softmax(selection, dim=-1)
        if log_softmax:
            return dist
        return torch.exp(dist)

    def state_lists_to_batch(self, state_lists):
        """
        Convert list of list states to batch for network

        param state_lists: 
            list of 'list[endoded states]'

        return 
            States to Tensor
        """
        assert isinstance(state_lists, list)
        batch_size = len(state_lists)
        batch = torch.zeros((batch_size,) + STATE_SHAPE, dtype=torch.float32)
        size = np.array(STATE_SHAPE).prod()
        for idx, state in enumerate(state_lists):
            decoded = torch.tensor([state_to_list(state, size)])
            batch[idx] = decoded.view(STATE_SHAPE)
        return batch.to(self.device)

    def valids_actions_to_tensor(self, valids_actions):
        mask = state_to_list(valids_actions, 111)
        return torch.tensor(mask, dtype=torch.bool).to(self.device)

    def remaining_pieces_to_tensor(self, remaining_pieces):
        mask = state_to_list(remaining_pieces, 55)
        return torch.tensor(mask, dtype=torch.bool).to(self.device)

    def train_batch(self, data):
        """
        Given a batch of training data, train the NN

        param data:
            list with training data

        return:
            Training loss
        """
        # data: [(state, p_target, v_target, b_target, valids_actions)]
        batch, p_targets, v_targets, c_targets, valids_actions = [], [], [], [], []
        for (state, p, v, c, actions) in data:
            # state and available_actions are encoded
            batch.append(state)
            p_targets.append(p)
            v_targets.append(v)
            c_targets.append(c)
            valids_actions.append(actions)
        batch = self.state_lists_to_batch(batch)

        self.train()
        self.optimizer.zero_grad()

        p_targets = [torch.tensor(p_target, dtype=torch.float32).to(self.device) for p_target in p_targets]     
        v_targets = torch.tensor(v_targets, dtype=torch.float32).to(self.device)
        c_targets = torch.tensor(c_targets, dtype=torch.float32).to(self.device)

        p_preds_t, v_preds, c_preds = self(batch)
        p_preds = []

        for i, a in enumerate(valids_actions):
            mask = self.valids_actions_to_tensor(a)
            p_preds.append(self.get_policy_value(p_preds_t[i], mask, True))

        # MSE
        loss_value = F.mse_loss(v_preds.squeeze(-1), v_targets)

        #cross entropy
        loss_policy = torch.zeros(1).to(self.device)
        for pred, target in zip(p_preds, p_targets):
            loss_policy += -torch.sum(pred * target)
        loss_policy = loss_policy / len(p_preds)

        # MSE
        loss_colab = F.mse_loss(c_preds, c_targets)

        loss = loss_policy + loss_value + loss_colab
        loss.backward()
        self.optimizer.step()

        # Return loss values to track total loss mean for epoch
        return (loss.item(), loss_policy.item(), loss_value.item(), loss_colab.item())

    def save(self, error_log, config, epoch, path, save_model, tag='latest', verbose=False):
        net_name = [f'AlphaZero_Dom_{tag}.ckpt', f'AlphaZero_Dom_model_{tag}.ckpt'][save_model]

        save_path = f'{path}/{self.save_path}'
        full_path = f'{save_path}{net_name}'

        if not os.path.exists(path):
            os.makedirs(path)

        if os.path.exists(full_path):
            # Save backup for tag
            os.rename(full_path, f'{save_path}{net_name[:-5]}_backup.ckpt')

        if save_model:
            torch.save({
                'model': self,
                'device': self.device,
                'error_log': error_log,
                'config': config,
                'epoch': epoch,
            }, full_path)
        else:
            torch.save({
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'device': self.device,
                'error_log': error_log,
                'config': config,
                'epoch': epoch,
                'residual_layers': self.residual_layers,
                'filters': self.filters,
            }, full_path)
        if verbose:
            print(f'Model saved with name: {net_name[:-5]}')
            print('Checkpoint saved')

    def load_checkpoint(self, save_path, tag='latest', load_logs=False, load_model=False):
        net_name = [f'AlphaZero_Dom_{tag}.ckpt', f'AlphaZero_Dom_model_{tag}.ckpt'][load_model]

        net_checkpoint = torch.load(f'{save_path}/{self.save_path}{net_name}')
        device = net_checkpoint['device']

        ret = [net_checkpoint['config']]
        model = None
        if load_model:
            ret.append(net_checkpoint['model'])
            model = ret[1]
        else:
            layers, filters = net_checkpoint['residual_layers'], net_checkpoint['filters']
            model = Net(layers, filters)
            model.load_state_dict(net_checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(net_checkpoint['optimizer_state_dict'])

        try:
            model = model.to(device)
            self.device = device
        except AssertionError:
            self.device = torch.device('cpu')
            model.to(self.device)

        # Load safe copy with right device
        if not load_model:
            self.load_state_dict(model.state_dict())

        if load_logs:
            ret.extend([net_checkpoint['error_log'], net_checkpoint['epoch']])
        return ret

    @staticmethod
    def load(path):
        net_checkpoint = torch.load(path)
        return net_checkpoint['model']