import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from ..utils import state_to_list

#STATE= [(56bits, 4bits, 2bits) x 41]
STATE_SHAPE = (1, 41, 62)
NUM_FILTERS = 256
KERNEL_SIZE = 3

class Net(nn.Module):
    """
    Neural Network for Alpha Zero implementation of Dominoes
    """
    def __init__(self, input_shape=STATE_SHAPE, policy_shape=111, device='cpu'):
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

        device = torch.device(device)
        self.device = device

        self.conv_in = nn.Sequential(
            nn.Conv2d(input_shape[0], NUM_FILTERS, kernel_size=KERNEL_SIZE, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.LeakyReLU(),
        ).to(device)

        # layers with residual
        self.conv_1 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=KERNEL_SIZE, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.LeakyReLU(),
            ).to(device)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=KERNEL_SIZE, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.LeakyReLU(),
            ).to(device)
        self.conv_3 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=KERNEL_SIZE, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.LeakyReLU(),
            ).to(device)
        self.conv_4 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=KERNEL_SIZE, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.LeakyReLU(),
            ).to(device)
        self.conv_5 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=KERNEL_SIZE, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.LeakyReLU(),
            ).to(device)
        self.conv_6 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=KERNEL_SIZE, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.LeakyReLU(),
            ).to(device)
        self.conv_7 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=KERNEL_SIZE, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.LeakyReLU(),
            ).to(device)
        self.conv_8 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=KERNEL_SIZE, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.LeakyReLU(),
            ).to(device)
        self.conv_9 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=KERNEL_SIZE, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.LeakyReLU(),
            ).to(device)
        self.conv_10 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=KERNEL_SIZE, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.LeakyReLU(),
            ).to(device)
        # self.conv_11 = nn.Sequential(
        #     nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=KERNEL_SIZE, padding=1),
        #     nn.BatchNorm2d(NUM_FILTERS),
        #     nn.LeakyReLU(),
        #     ).to(device)
        # self.conv_12 = nn.Sequential(
        #     nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=KERNEL_SIZE, padding=1),
        #     nn.BatchNorm2d(NUM_FILTERS),
        #     nn.LeakyReLU(),
        #     ).to(device)
        # self.conv_13 = nn.Sequential(
        #     nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=KERNEL_SIZE, padding=1),
        #     nn.BatchNorm2d(NUM_FILTERS),
        #     nn.LeakyReLU(),
        #     ).to(device)
        # self.conv_14 = nn.Sequential(
        #     nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=KERNEL_SIZE, padding=1),
        #     nn.BatchNorm2d(NUM_FILTERS),
        #     nn.LeakyReLU(),
        #     ).to(device)
        # self.conv_15 = nn.Sequential(
        #     nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=KERNEL_SIZE, padding=1),
        #     nn.BatchNorm2d(NUM_FILTERS),
        #     nn.LeakyReLU(),
        #     ).to(device)
        # self.conv_16 = nn.Sequential(
        #     nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=KERNEL_SIZE, padding=1),
        #     nn.BatchNorm2d(NUM_FILTERS),
        #     nn.LeakyReLU(),
        #     ).to(device)
        # self.conv_17 = nn.Sequential(
        #     nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=KERNEL_SIZE, padding=1),
        #     nn.BatchNorm2d(NUM_FILTERS),
        #     nn.LeakyReLU(),
        #     ).to(device)
        # self.conv_18 = nn.Sequential(
        #     nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=KERNEL_SIZE, padding=1),
        #     nn.BatchNorm2d(NUM_FILTERS),
        #     nn.LeakyReLU(),
        #     ).to(device)
        # self.conv_19 = nn.Sequential(
        #     nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=KERNEL_SIZE, padding=1),
        #     nn.BatchNorm2d(NUM_FILTERS),
        #     nn.LeakyReLU(),
        #     ).to(device)

        # value head
        self.conv_val = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
        ).to(device)

        body_out_shape = (NUM_FILTERS, ) + input_shape[1:]
        conv_val_size = self._get_conv_val_size(body_out_shape)

        self.value = nn.Sequential(
            nn.Linear(conv_val_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        ).to(device)

        # policy head
        self.conv_policy = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, input_shape[0], kernel_size=1),
            nn.BatchNorm2d(input_shape[0]),
            nn.LeakyReLU(),
        ).to(device)
        conv_policy_size = self._get_conv_policy_size(body_out_shape)
        self.policy = nn.Sequential(
            nn.Linear(conv_policy_size, policy_shape)
        ).to(device)

        #optimizer
        self.optimizer = optim.SGD(self.parameters(), lr=0.2, momentum=0.9, weight_decay=1e-4)


    def _get_conv_val_size(self, shape):
        o = self.conv_val(torch.zeros(1, *shape).to(self.device))
        return int(np.prod(o.size()))

    def _get_conv_policy_size(self, shape):
        o = self.conv_policy(torch.zeros(1, *shape).to(self.device))
        return int(np.prod(o.size()))

    def forward(self, x):
        batch_size = x.size()[0]

        v = self.conv_in(x)
        
        v = self.conv_1(v)
        v = self.conv_2(v)
        v = self.conv_3(v)
        v = self.conv_4(v)
        v = self.conv_5(v)
        v = self.conv_6(v)
        v = self.conv_7(v)
        v = self.conv_8(v)
        v = self.conv_9(v)
        v = self.conv_10(v)
        # v = self.conv_11(v)
        # v = self.conv_12(v)
        # v = self.conv_13(v)
        # v = self.conv_14(v)
        # v = self.conv_15(v)
        # v = self.conv_16(v)
        # v = self.conv_17(v)
        # v = self.conv_18(v)
        # v = self.conv_19(v)
        
        val = self.conv_val(v)
        val = self.value(val.view(batch_size, -1))

        pol = self.conv_policy(v)
        pol = self.policy(pol.view(batch_size, -1))

        return pol, val

    def predict(self, s, valids_actions):
        """
        Infer node data given an state

        param s: 
            list of encoded states of the game
        param available_actions
            list of encoded valids actions from a position of the game

        return
            (Move probabilities P vector, value V vector)
        """
        self.eval()
        batch = self.state_lists_to_batch(s)
        masks = [self.valids_actions_to_tensor(va) for va in valids_actions]
        pol, val = self(batch)
        pol = [self.get_policy_value(p, mask, False) for p, mask in zip(pol, masks)]
        return pol, val

    def get_policy_value(self, logits, mask, log_softmax):
        """
        Get move probabilities distribution.

        param logits:
            list of 111 bits. Raw policy head
        param available_actions
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

    def train_batch(self, data):
        """
        Given a batch of training data, train the NN

        param data:
            list with training data

        return:
            Training loss
        """
        # data: [(state, p_target, v_target, valids_actions)]
        batch, p_targets, v_targets, valids_actions = [], [], [], []
        for (state, p, v, actions) in data:
            # state and available_actions are encoded
            batch.append([self.state_lists_to_batch(state)])
            p_targets.append(p)
            v_targets.append(v)
            valids_actions.append(actions)

        self.train()
        self.optimizer.zero_grad()

        p_targets = torch.FloatTensor(p_targets).to(self.device)    
        v_targets = torch.FloatTensor(v_targets).to(self.device)
        p_preds, v_preds = self(batch)

        for i, a in enumerate(valids_actions):
            mask = self.valids_actions_to_tensor(a)
            p_preds[i] = self.get_policy_value(p_preds[i], mask, True)

        loss_value = F.mse_loss(v_preds.squeeze(-1), v_targets)
        loss_policy = -torch.sum(p_preds * p_targets)

        loss = loss_policy + loss_value
        loss.backward()
        self.optimizer.step()

        # Return loss values to track total loss mean for epoch
        return (loss.item(), loss_policy.item(), loss_value.item())

    def save(self, error_log, epoch, save_path, tag='latest', verbose=False):
        net_name = f'AlphaZero_Dom_{tag}_.ckpt'
        save_path = f'{save_path}/{self.save_path}'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if os.path.exists(save_path + net_name):
            # Save backup for tag
            latest_model = torch.load(save_path + net_name)
            torch.save(latest_model, f'{save_path}AlphaZero_Dom_backup-{tag}_.ckpt')

        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'device': self.device,
            'error_log': error_log,
            'epoch': epoch,
        }, save_path + net_name)
        if verbose:
            print(f'Model saved with name: {net_name[:-5]}')

    def load(self, save_path, tag='latest', load_logs=False):
        net_name = f'AlphaZero_Dom_{tag}.ckpt'
        net_checkpoint = torch.load(save_path + '/' + self.save_path + net_name)
        device = net_checkpoint['device']
        self.load_state_dict(net_checkpoint['model_state_dict'])
        self.to(device)
        self.device = device
        self.optimizer.load_state_dict(net_checkpoint['optimizer_state_dict'])
        if load_logs:
            return net_checkpoint['error_log'], net_checkpoint['epoch']
