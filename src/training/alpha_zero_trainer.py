from multiprocessing import Pool
from torch.utils.tensorboard import SummaryWriter
from ..models import alpha_zero_net as Net
from .trainer import Trainer
from ..games import *

import random
import time
import os
import json
import torch.profiler

class AlphaZeroTrainer(Trainer):
    """
    Trainer manager for Alpha Zero model
    """
    def __init__(
        self,
        net: Net,
        batch_size: int,
        handouts: int,
        rollouts: int,
        max_number: int,
        pieces_per_player: int,
        tau_threshold: int = 6,
        data_path: str = 'data'
    ):
        """
        param net: nn.Module
            Neural Network to train
        param batch_size: int
            Size of training data used per epoch
        param handouts: int
            Number of handouts per move search
        param rollouts: int
            Number of rollouts per handout
        param max_number:
            Max piece number
        param pieces_per_player:
            Number of pieces distributed per player
        param tau_threshold:
            Threshold for temperature behavior to become equivalent to argmax
        param data_path: string
            Path to the folder where training data will be saved
        """
        self.net = net
        self.batch_size = batch_size
        self.handouts = handouts
        self.rollouts = rollouts
        self.max_number = max_number
        self.pieces_per_player = pieces_per_player
        self.tau_threshold = tau_threshold
        self.data_path = data_path
        self.error_log = []

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        
    def self_play(
        self,
        handouts,
        rollouts
    ):
        """
        Simulate one game via self play and save moves played

        param handouts: int
            Number of handouts per move search
        param rollouts: int
            Number of rollouts per handout

        return
            Data of the game. List[(state, pi, result)]
        """
        data = []
        game_over = False
        root = True
        players = [BasePlayer(i) for i in range(4)]
        manager = DominoManager()
        manager.init(players, hand_out, self.max_number, self.pieces_per_player)

        while not game_over:
            stats = {}
            cur_player = players[manager.domino.current_player]
            selector = selector_maker(stats, cur_player.valid_moves(), cur_player.pieces_per_player - len(cur_player.pieces), root, self.tau_threshold)
            encoder = encoder_generator(self.max_number)
            rollout = rollout_maker(stats, self.net)

            root = False

            state, action, pi = monte_carlo(
                cur_player, 
                encoder, 
                rollout, 
                selector,
                handouts,
                rollouts,
            )
            _, mask = get_valids_data(manager.domino)
            game_over = manager.step(True, action)
            data.append((state, pi, cur_player, mask))

        training_data = []
        for state, pi, player, mask in data:
            end_value = [0, 0, 0]
            end_value[player.team] = 1
            end_value[1 - player.team] = -1
            result = end_value[manager.domino.winner] 
            training_data.append((state, pi, result, mask))
        return data

    def policy_iteration(
        self,
        epoch: int,
        simulate: bool,
        num_process=1,
        verbose=False,
        save_data=False
    ):
        """
        Do a training iteration (epoch)

        param epoch: int
            Current epoch
        param simulate: bool
            Set True if data must be generated via self play. Set False to load saved data if possible. 
            If there is not enough saved data, missing data will be generated via self play.
        param num_process: int
            Number of parallel self_play games. Set to one to disable parallelism.
        param verbose: bool
            Set True to enable verbose log
        param save_data: bool
            Set True to save generated training data

        return
            (total_loss, loss_policy, loss_value)
        """
        if verbose:
            print('Getting training data...')
        data = self._get_data(simulate, num_process, verbose, self.batch_size)
        
        if save_data:
            num_files = len([name for name in os.listdir(self.data_path) if os.path.isfile(name)])
            path = f'{self.data_path}/training_data_{num_files}'
            with open(path, 'w') as writer:
                json.dump(data, writer)
            if verbose:
                print(f'Training data saved at {path}')

        batch = random.sample(data, self.batch_size)
        if verbose:
            print(f'[Epoch {epoch}] -- Training net --')
            start = time.time()

        Trainer.adjust_learning_rate(epoch, self.net.optimizer)
        loss = self.net.train_batch(batch)
        self.error_log.append(loss)
        self.net.save(self.error_log, epoch, verbose=True)

        if verbose:
            print(f'-- Training took {str(int(time.time() - start))} seconds --')
            print(f'policy head loss: {loss[1]} -- value head loss: {loss[2]} -- TOTAL LOSS: {loss[0]}')
            print('Checkpoint saved')
            print('')

        return loss

    def _get_data(
        self,
        simulate: bool,
        num_process: int,
        verbose: bool,
        batch_size: int
    ):
        data = []
        num_games = 0

        if simulate:
            if verbose:
                print('Simulations started')
                start = time.time()

            if num_process > 1:
                while len(data) < batch_size:
                    jobs = [self.handouts, self.rollouts] * (batch_size / num_process)
                    num_games += batch_size / num_process
                    pool = Pool(num_process)
                    new_data = pool.map(self.self_play, jobs)
                    pool.close()
                    pool.join()
                    for d in new_data:
                        data.extend(d)
            else:
                while len(data) < batch_size:
                    data.extend(self.self_play(self.handouts, self.rollouts))
                    num_games += 1

            if verbose:
                print(f'Simulated {num_games} in {str(int(time.time() - start))} seconds')
        else:
            # Get saved training data
            if verbose:
                print('Loading saved data...')
            file_names = [name for name in os.listdir(self.data_path) if os.path.isfile(name)]
            file_names.sort()
            file_names.reverse() # Reverse to get newest data first

            for name in file_names:
                path = f'{self.data_path}/{name}'

                with open(path, 'r') as reader:
                    data.extend(json.load(reader))

                if verbose:
                    print(f'File {name} loaded. Current batch_size: {len(data)}')

                if len(data) >= batch_size:
                    break
            if len(data) < batch_size:
                if verbose:
                    print('Insuficient data. Proceding to generate data with self play')
                data.extend(self._get_data(True, num_process, verbose, batch_size - len(data)))

        return data

    def train(self, epochs: int, simulate: bool, load_checkpoint: bool, tag='latest', num_process=1, verbose=False, save_data=False):
        """
        Training Pipeline
        """
        writer = SummaryWriter()
        last_epoch = -1

        if load_checkpoint:
            error_log, e = self.net.load(tag, True)
            self.error_log = error_log
            last_epoch = e

        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=epochs, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./runs/'),
            record_shapes=True,
            with_stack=True,
            profile_memory=True
        ) as prof:
            for e in range(last_epoch + 1, epochs + 1):
                loss = self.policy_iteration(e, simulate, num_process, verbose, save_data)
                total_loss, policy_loss, value_loss = loss
                if simulate:
                    simulate = False
                loss = {
                    'Total loss': total_loss,
                    'Policy loss': policy_loss,
                    'Value loss': value_loss,
                }
                writer.add_scalars('Loss', loss, e)
                prof.step()

        writer.flush()
