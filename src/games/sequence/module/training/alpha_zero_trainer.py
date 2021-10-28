from typing import List, Any
from torch.multiprocessing import Pool, set_start_method
from torch.utils.tensorboard import SummaryWriter
from ..players import AlphaZeroNet, alphazero_utils as utils, mc_utils, BasePlayer, handout
from .trainer import Trainer
from ..sequence import Sequence

import random
import time
import os
import json
import torch
import shutil
import ray

class AlphaZeroTrainer(Trainer):
    """
    Trainer manager for Alpha Zero Sequence model
    """
    def __init__(
        self,
        batch_size: int,
        handouts: int,
        rollouts: int,
        alpha: float,
        number_of_players: int,
        cards_per_player: int,
        players_colors: List[Any],
        data_path: str,
        save_path: str,
        lr: int,
        cput: int,
        residual_layers: int,
        num_filters: int,
        tau_threshold: int = 8,
    ):
        """
        param batch_size: int
            Size of training data used per epoch
        param handouts: int
            Number of handouts per move search
        param rollouts: int
            Number of rollouts per handout
        param alpha: float
            Parameter of Dirichlet random variable
        param number_of_players: int:
            Number of players in the game
        param cards_per_player: int:
            Number of cards distributed per player
        param players_colors: List[Any]
            Color of every player
        param data_path: string
            Path to the folder where training data will be saved
        param save_path: string
            Path to the folder where network data will be saved    
        param lr: int
            Learning rate
        param cput: int
            Exploration constant
        param residual_layers: int
            Number of residual convolutional layers
        param num_filters: int
            Number of channels of residual convolutional layers
        param tau_threshold: int
            Threshold for temperature behavior to become equivalent to argmax
        """
        self.batch_size = batch_size
        self.handouts = handouts
        self.rollouts = rollouts
        self.alpha = alpha
        self.number_of_players = number_of_players
        self.cards_per_player = cards_per_player
        self.players_colors = players_colors
        self.data_path = data_path
        self.save_path = save_path
        self.lr = lr
        self.cput = cput
        self.tau_threshold = tau_threshold
        self.error_log = []

        self.net = AlphaZeroNet(residual_layers, num_filters, lr=lr)
        device = 'cpu'
        if torch.cuda.is_available():
            device = "cuda:0"
            if torch.cuda.device_count() > 1:
                self.net = torch.nn.DataParallel(self.net)

        device = torch.device(device)
        self.net.to(device)
        self.net.device = device

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        
        try:
            set_start_method('spawn')
        except RuntimeError:
            pass

    def self_play_fractional(self, num_gpus):
        @ray.remote(num_gpus=num_gpus, max_calls=1)
        def _self_play_fractional(args):
            return self.self_play(args)

        return _self_play_fractional

    def self_play(
        self,
        args,
    ):
        """
        Simulate one game via self play and save moves played

        param handouts: int
            Number of handouts per move search
        param rollouts: int
            Number of rollouts per handout
        param alpha: float
            Parameter of Dirichlet random variable

        return
            Data of the game. List[(state, pi, result)]
        """
        handouts, rollouts, alpha = args
        data = []
        game_over = False
        root = True
        sequence = Sequence()
        sequence.reset(handout, self.number_of_players, self.players_colors, self.cards_per_player)
        turn = 1

        while not game_over:
            stats = {}
            cur_player = BasePlayer.from_sequence(sequence)
            selector = utils.selector_maker(stats, cur_player.valid_moves(), turn, root, self.tau_threshold, alpha)
            encoder = utils.encode
            rollout = utils.rollout_maker(stats, self.net, self.cput)

            root = False

            state, action, pi = mc_utils.monte_carlo(
                cur_player, 
                encoder, 
                rollout, 
                selector,
                handouts,
                rollouts,
                self.net,
            )
            #Get valids actions mask
            valids = sequence._valid_moves()
            mask = utils.encode_valids(valids)

            #creating partner's hand entry
            [partner, *_] = list(sequence.partners) # Should return just one player, *_ added for safety in games with teams size greater than 2
            partner_hand = sequence.players[partner].cards
            pieces_mask = 0
            for p in partner_hand:
                pieces_mask += utils.table_bit(*p)
            partner_hand = utils.state_to_list(pieces_mask, 104)

            game_over = sequence.step(action)
            turn += 1
            data.append((state, pi.tolist(), cur_player, mask, partner_hand))

        training_data = []
        for state, pi, player, mask, partner_hand in data:
            #//TODO: Change this for more than 2 teams
            end_value = [0, 0, 0]
            end_value[player.team] = 1
            end_value[1 - player.team] = -1
            result = end_value[sequence.winner] 
            training_data.append((state, pi, result, partner_hand, mask))
        return training_data

    def policy_iteration(
        self,
        epoch: int,
        simulate: bool, 
        sample: int,
        tag: str,
        num_process=1,
        num_gpus=1,
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
        param sample: int:
            Size of NN input in training
        param tag: str
            Tag of the checkpoint to be saved
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
        data = self._get_data(simulate, num_process, num_gpus, verbose, self.batch_size)
        
        if save_data:
            num_files = len([name for name in os.listdir(self.data_path) if os.path.isfile(f'{self.data_path}/{name}')])
            path = f'{self.data_path}/training_data_{num_files}.json'
            with open(path, 'w') as writer:
                json.dump(data, writer)
            if verbose:
                print(f'Training data saved at {path}')

        if verbose:
            print(f'[Epoch {epoch}] -- Training net --')
            start = time.time()

        self.adjust_learning_rate(epoch, self.net.optimizer)
        total_loss, policy_loss, value_loss = 0,0,0
        batch_size = len(data)
        total = 0

        for _ in range(batch_size // sample):
            batch = random.sample(data, sample)
            total += sample
            loss = self.net.train_batch(batch)
            total_loss += loss[0]
            policy_loss += loss[1]
            value_loss += loss[2]
        
        loss = (total_loss / total, policy_loss / total, value_loss / total)
        self.error_log.append(loss)

        config = self.build_config(sample, tag, epoch)
        if self.loss > loss[0]:
            self.loss = loss[0]
            config = self.build_config(sample, tag, epoch)
            self.net.save(self.error_log, config, epoch, self.save_path, True, tag + '-min', verbose=True)
            self.net.save(self.error_log, config, epoch, self.save_path, False, tag, verbose=True)

        if verbose:
            print(f'-- Training took {str(int(time.time() - start))} seconds --')
            print(f'policy head loss: {loss[1]} -- value head loss: {loss[2]} -- TOTAL LOSS: {loss[0]}')
            print('')

        return loss

    def _get_data(
        self,
        simulate: bool,
        num_process: int,
        num_gpus: int,
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
                    jobs = [(self.handouts, self.rollouts, self.alpha)] * (batch_size // num_process)
                    num_games += batch_size // num_process
                    pool = Pool(num_process)
                    new_data = pool.map(self.self_play, jobs)
                    pool.close()
                    pool.join()
                    for d in new_data:
                        data.extend(d)
            elif num_gpus < 1:
                while len(data) < batch_size:
                    args = [(self.handouts, self.rollouts, self.alpha)]
                    num_games += batch_size // num_gpus
                    ray.get([self.self_play_fractional(num_gpus).remote(args) for _ in range(int(1 / num_gpus))])
            else:
                while len(data) < batch_size:
                    data.extend(self.self_play((self.handouts, self.rollouts, self.alpha)))
                    num_games += 1

            if verbose:
                print(f'Simulated {num_games} games in {str(int(time.time() - start))} seconds')
        else:
            # Get saved training data
            if verbose:
                print('Loading saved data...')
            file_names = [name for name in os.listdir(self.data_path) if os.path.isfile(f'{self.data_path}/{name}')]
            file_names.sort(key= lambda x: int(x[14:-5]), reverse=True)

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
                data.extend(self._get_data(True, num_process, num_gpus, verbose, batch_size - len(data)))

        return data

    def train(
        self,
        epochs: int,
        simulate: bool,
        load_checkpoint: bool,
        load_model: bool,
        sample: int,
        tag='latest',
        num_process=1,
        num_gpus=1,
        verbose=False,
        save_data=False
    ):
        """
        Training Pipeline
        """
        if num_gpus < 1:
            ray.init(num_cpus=4, num_gpus=num_gpus)

        writer = SummaryWriter(comment=tag)
        last_epoch = 1
        self.loss = 1e30
        self.epochs = epochs

        if load_checkpoint:
            last_epoch, sample, tag = self.load_checkpoint(load_model, tag, writer, epochs, verbose)
        
        for e in range(last_epoch, self.epochs + 1):
            loss = self.policy_iteration(e, simulate, sample, tag, num_process, num_gpus, verbose, save_data)
            simulate = True
            save_data = True
            self.write_loss(writer, e - 1, *loss)

        config = self.build_config(sample, tag, self.epochs)
        self.net.save(self.error_log, config, self.epochs, self.save_path, True, tag + '-completed', verbose=True)
        writer.flush()

    def load_checkpoint(self, load_model, tag, writer, epochs, verbose):
        if load_model:
            config, model, error_log, e = self.net.load_checkpoint(self.save_path, tag, True, load_model)
            if self.net.device == "cuda" and torch.cuda.is_available():
                model.device = "cuda:0"
                if torch.cuda.device_count() > 1:
                    model = torch.nn.DataParallel(model)
                model.to(model.device)
            self.net = model
        else:
            config, error_log, e = self.net.load_checkpoint(self.save_path, tag, True)
        
        if verbose:
            print(json.dumps(config, indent=4))

        self.remove_last_run(tag)
        self.error_log = error_log
        for ep, loss in enumerate(error_log):
            self.write_loss(writer, ep, *loss)
        writer.flush()

        last_epoch = e + 1
        self.epochs += e
        sample, tag = self.load_config(config, epochs)
        return last_epoch, sample, tag


    def build_config(self, sample, tag, cur_epoch):
        return {
            "batch_size": self.batch_size,
            "handouts": self.handouts,
            "rollouts": self.rollouts,
            "tau_threshold": self.tau_threshold,
            "number_of_players": self.number_of_players,
            "cards_per_player": self.cards_per_player,
            "players_colors": self.players_colors,
            "data_path": self.data_path, 
            "save_path": self.save_path,
            "lr": self.lr,
            "cput": self.cput,

            "min_loss": self.loss,
            "epochs": self.epochs - cur_epoch,
            "sample": sample,
            "tag": tag,
        }

    def load_config(self, config, epochs: int = 0):
        self.batch_size = config["batch_size"]
        self.handouts = config["handouts"]
        self.rollouts = config["rollouts"]
        self.tau_threshold = config["tau_threshold"]
        self.number_of_players = config["number_of_players"]
        self.cards_per_player = config["cards_per_player"]
        self.players_colors = config["players_colors"]
        self.data_path = config["data_path"]
        self.save_path = config["save_path"]
        self.lr = config["lr"]
        self.cput = config["cput"]
        self.loss = config["min_loss"]
        self.epochs += config["epochs"]
        if epochs:
            self.epochs += epochs
        sample = config["sample"]
        tag = config["tag"]

        return sample, tag

    def write_loss(self, writer, e, total, policy, value):
        loss = {
            'Total loss': total,
            'Policy loss': policy,
            'Value loss': value,
            }
        writer.add_scalars('Loss', loss, e + 1)

    def adjust_learning_rate(self, epoch, optimizer):
        lr = self.lr

        if epoch == 150:
            self.lr = lr / 10
        elif epoch == 100:
            self.lr = lr / 10
        elif epoch == 50:
            self.lr = lr / 10

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

    def remove_last_run(self, tag, path='./runs/'):
        if os.path.exists(path):
            folder_names = [name for name in os.listdir(path) if tag in name]
            try:
                for f in folder_names:
                    shutil.rmtree(path + f)
            except [OSError, Exception] as e:
                print(e)
