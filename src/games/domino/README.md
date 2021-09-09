# Dominoes
Implementation of the [dominoes](https://en.wikipedia.org/wiki/Dominoes) game with support for the 6x6 and 9x9 versions and clever agents, including [Alpha Zero implementation for Dominoes](#cooperaitive-agent)

- [Dominoes](#dominoes)
  - [Dependencies](#dependencies)
  - [Vocabulary](#vocabulary)
  - [Execution](#execution)
    - [Info mode](#info-mode)
    - [Play mode](#play-mode)
    - [Match mode](#match-mode)
  - [Tournaments](#tournaments)
  - [CooperAItive agent](#cooperaitive-agent)
    - [Train](#train)
  - [Development](#development)
    - [Players](#players)
    - [Rules](#rules)
    - [Hands](#hands)
    - [Behaviors](#behaviors)
## Dependencies
To able to run the domino game you will need to have [python](https://www.python.org/) installed.
To install the requirements run the following command:
```bash
pip install -r requirements.txt
```

## Vocabulary
- `Player`: An implementation of a domino player strategy (e.g Random, MCTS, Alpha Zero, Botagorda, etc)
- `Rule`: An implementation of a domino game mode. (e.g Single Game, 100 points, Two of Three)
- `Hands`: Implementations of special pieces handouts.
- `Behaviors`: Implementations of different player modifiers for special game situations, like a special move in starts.

## Execution
A set of different modes to run domino plays. See the help command for listing them all.
```bash
python domino.py --help
```

### Info mode
All the game information that you need to know to run custom games is available using the command:
```bash
python domino.py info
```
The output of this command show all the players, rules, hands and behaviors implemented.

### Play mode
To run and play the game you can use the following command:
```bash
python domino.py play
```
This command runs a default game between `Random` players.
To know how to run custom plays refer to the command help:
```bash
python domino.py play --help
```
### Match mode
This mode is meant to make a match between player and a list of opponents. See the command help for usage details.
```bash
python domino.py match --help
```

## Tournaments
There are some interesting tournaments implemented that can be executed with different configurations.
To list the available tournaments use:
```bash
python tournaments.py --help
```
And then use the help of each tournament mode to know their usage like in the previous section.

## CooperAItive agent
The only implementation so far of an "intelligent" agent for dominoes (9x9 version), is [Alpha Zero](https://en.wikipedia.org/wiki/AlphaZero). The code for it is located in:

- MCTS with UCT (Monte Carlo Tree Search with Upper Confidence Bound): 
  - `module/players/strategies/alphazero.py` (`Player`)
  - `module/players/strategies/utils/alphazero.py`
  - `module/players/strategies/utils/mc.py`
- Model for the Deep Convolutional Neural Network:
  - `module/players/strategies/models/alpha_zero_model.py`
- Training controller:
  - `module/training/alpha_zero_trainer.py`

> The Neural Network is implemented using [pytorch](https://pytorch.org)

The identifier for this player in the command arguments is `AlphaZero`.
### Train
You can train your own version of Alpha Zero and experiment with the hyperparameters. Training supports checkpoints, verbose mode and TensorBoard integration.

To train the cooperAItive agent use:
```bash
python train.py config
```
Where config is the path to the training configuration file.
That configuration file should be a json with the following fields:
```json
{
    "batch_size": int,        
    "handouts": int,          
    "rollouts": int,          
    "alpha": float,           
    "max_number": int,        
    "pieces_per_player": int, 
    "data_path": str,          
    "save_path": str,         
    "lr": float,              
    "tau_threshold": int,     
    "epochs": int,            
    "simulate": bool,         
    "load_checkpoint": bool,  
    "load_model": bool,       
    "sample": any,            
    "tag": str,               
    "num_process": int,       
    "verbose": bool,          
    "save_data": bool,        
}
```

- `batch_size`: Size of the batch used in a training iteration.
- `handouts`: Number of random handouts used to build a distribution of player pieces in MCTS.
- `rollouts`: Number of playouts or searches in a MCTS iteration per handout.
- `alpha`: Parameter of the Dirichlet Random Variable used by Alpha Zero.
- `max_number`: Max piece number.
- `pieces_per_player`: Number of pieces distributed per player.
- `data_path`: Path to the folder where training data will be saved.
- `save_path`: Path to the folder where network data will be saved .
- `lr`: Learning Rate.
- `tau_threshold`: Threshold for temperature behavior to become equivalent to argmax.
- `epochs`: Number of training epochs.
- `simulate`: Set True if data must be generated via self play. Set False to load saved data if possible. If there is not enough saved data, missing data will be generated via self play.
- `load_checkpoint`: Set True to load last training checkpoint with tag provided.
- `load_model`: Set True to load the latest model saved with tag provided. 
- `sample`: Size of training input that will be extracted from training batch.
- `tag`: Tag to identify trained model or training checkpoints.
- `num_process`: If this value is greater than one, this value will represent the number of parallel self plays used to generate training data. Do not use this with CUDA.
- `verbose`: Set verbose mode.
- `save_data`: Set False to avoid re-saving loaded data.

To start tensorboard run:
```bash
tensorboard --logdir='.'
```
This will search for all `*.tfevents.*` located in actual folder or subfolders recursively. If training is setup with the above instruccion, then `*.tfevents.*` will be in `./runs` folder.
## Development
You can add your own `player`, `rules`, `hands` and `behaviors` to the game.

### Players 
To add player create a file `module/players/strategies/my_custom_player.py` with a class that inherits from the `BasePlayer` defined in `module/players/player.py`.

Example:
```python
from ..player import BasePlayer

class MyCustomPlayer(BasePlayer):
    ...

    def filter(self, valids=None):
        return self.valid_moves() if valids is None else valids
```

To make your player available in the execution commands you need to register the player in the `PLAYERS` list that you can find in `module/players/strategies/__init__.py`

```python
...
from .my_custom_player import MyCustomPlayer

PLAYERS = [
    ...,
    MyCustomPlayer,
]
```

### Rules
Same as the previous section, create a file `module/rules/my_rule.py`, inherit from `BaseRule` located at `module/rules/rules.py` and register your rule in the `RULES` list that you can find in `module/rules/__init__.py`

### Hands
To add a hand create a file `module/players/strategies/hands/my_hand.py`. 

Hands are functions that receive the game the `max_number` and the `pieces_per_player` of the game an return a list of 4 `PlayerView` objects

See the [hand_out](./module/players/hands/hand_out.py) file for reference.

You will need to register your `hand` in the `HAND` list of 
`module/players/strategies/hands/__init__.py`. 

### Behaviors
The behaviors are player implementations under the `module/players/behaviors/` folder and registered in the `BEHAVIORS` list of `module/players/behaviors/__init__.py`

See the `player` section above for reference. 